import torch
from torch import nn as nn
import numpy as np
import isegm.ritm.model.initializer as initializer


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False,
        )

        self.apply(
            initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups)
        )


# class DistMaps(nn.Module):
#     def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False, dynamic_radius=False):
#         super(DistMaps, self).__init__()
#         self.spatial_scale = spatial_scale
#         self.norm_radius = norm_radius
#         self.cpu_mode = cpu_mode
#         self.use_disks = use_disks
#         self.dynamic_radius = dynamic_radius
#         if self.cpu_mode:
#             from isegm.utils.cython import get_dist_maps

#             self._get_dist_maps = get_dist_maps

#     def get_coord_features(self, points, batchsize, rows, cols):
#         if self.cpu_mode:
#             coords = []
#             for i in range(batchsize):
#                 norm_delimeter = (
#                     1.0 if self.use_disks else self.spatial_scale * self.norm_radius
#                 )
#                 coords.append(
#                     self._get_dist_maps(
#                         points[i].cpu().float().numpy(), rows, cols, norm_delimeter
#                     )
#                 )
#             coords = (
#                 torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
#             )
#         else:
#             num_points = points.shape[1] // 2
#             points = points.view(-1, points.size(2))
#             points, points_order = torch.split(points, [2, 1], dim=1)

#             invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
#             row_array = torch.arange(
#                 start=0, end=rows, step=1, dtype=torch.float32, device=points.device
#             )
#             col_array = torch.arange(
#                 start=0, end=cols, step=1, dtype=torch.float32, device=points.device
#             )

#             coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
#             coords = (
#                 torch.stack((coord_rows, coord_cols), dim=0)
#                 .unsqueeze(0)
#                 .repeat(points.size(0), 1, 1, 1)
#             )

#             add_xy = (points * self.spatial_scale).view(
#                 points.size(0), points.size(1), 1, 1
#             )
#             coords.add_(-add_xy)
#             if not self.use_disks:
#                 coords.div_(self.norm_radius * self.spatial_scale)
#             coords.mul_(coords)

#             coords[:, 0] += coords[:, 1]
#             coords = coords[:, :1]

#             coords[invalid_points, :, :, :] = 1e6

#             coords = coords.view(-1, num_points, 1, rows, cols)
#             coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
#             coords = coords.view(-1, 2, rows, cols)

#         if self.use_disks:
#             coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
#         else:
#             coords.sqrt_().mul_(2).tanh_()

#         return coords

#     def forward(self, x, coords):
#         return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])
class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False, dynamic_radius=False, overwrite_maps=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        self.dynamic_radius = dynamic_radius
        self.overwrite_maps = overwrite_maps
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps

            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.dynamic_radius:
            return self.get_dynamic_disks(points, batchsize, rows, cols)
        
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = (
                    1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                )
                coords.append(
                    self._get_dist_maps(
                        points[i].cpu().float().numpy(), rows, cols, norm_delimeter
                    )
                )
            coords = (
                torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
            )
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order, _= torch.split(points, [2, 1, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(
                start=0, end=rows, step=1, dtype=torch.float32, device=points.device
            )
            col_array = torch.arange(
                start=0, end=cols, step=1, dtype=torch.float32, device=points.device
            )

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = (
                torch.stack((coord_rows, coord_cols), dim=0)
                .unsqueeze(0)
                .repeat(points.size(0), 1, 1, 1)
            )

            add_xy = (points * self.spatial_scale).view(
                points.size(0), points.size(1), 1, 1
            )
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords
    
    # def get_dynamic_disks(self, points, batchsize, rows, cols):
    #     half_num_points = points.shape[1] // 2
    #     maps = torch.zeros((batchsize, 2, rows, cols), device=points.device)

    #     x_grid, y_grid = torch.meshgrid(torch.arange(rows, device=points.device), 
    #                                     torch.arange(cols, device=points.device), indexing='ij')

    #     for batch_idx in range(batchsize):
    #         for map_type in range(2):  # 0 for positive clicks, 1 for negative clicks
    #             for point_idx in range(half_num_points):
    #                 point = points[batch_idx, point_idx + (half_num_points * map_type)]
    #                 x, y, _, radius = point
    #                 if x < 0 or y < 0:  # Skip invalid points
    #                     continue

    #                 int_radius = int(torch.ceil(radius))
    #                 x_min, x_max = max(0, int(x) - int_radius), min(rows, int(x) + int_radius + 1)
    #                 y_min, y_max = max(0, int(y) - int_radius), min(cols, int(y) + int_radius + 1)

    #                 mask = (x_grid[x_min:x_max, y_min:y_max] - x)**2 + (y_grid[x_min:x_max, y_min:y_max] - y)**2 <= radius**2
    #                 maps[batch_idx, map_type, x_min:x_max, y_min:y_max][mask] = 1

    #     return maps.float()
    
    def get_dynamic_disks(self, points, batchsize, rows, cols):
        half_num_points = points.shape[1] // 2
        
        maps = torch.zeros((batchsize, 2, rows, cols), device=points.device)

        x_grid, y_grid = torch.meshgrid(torch.arange(rows, device=points.device), 
                                        torch.arange(cols, device=points.device), indexing='ij')

        for batch_idx in range(batchsize):
            
            # False - positive map, True - negative map
            points_with_maps = [(point, i >= half_num_points) for i, point in enumerate(points[batch_idx])]
        
            #filter out invalid clicks and sort
            points_with_maps = [point_map_pair for point_map_pair in points_with_maps if point_map_pair[0][2] > -1]
            points_with_maps = sorted(points_with_maps, key=lambda pair: pair[0][2])
            
            for point, map_idx in points_with_maps:
                x, y, _, radius = point
                
                int_radius = int(torch.ceil(radius))
                x_min, x_max = max(0, int(x) - int_radius), min(rows, int(x) + int_radius + 1)
                y_min, y_max = max(0, int(y) - int_radius), min(cols, int(y) + int_radius + 1)
                
                mask = (x_grid[x_min:x_max, y_min:y_max] - x)**2 + (y_grid[x_min:x_max, y_min:y_max] - y)**2 <= radius**2
                maps[batch_idx, int(map_idx), x_min:x_max, y_min:y_max][mask] = 1
                
                if self.overwrite_maps:
                    maps[batch_idx, int(not map_idx), x_min:x_max, y_min:y_max][mask] = 0

        return maps.float()

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor
