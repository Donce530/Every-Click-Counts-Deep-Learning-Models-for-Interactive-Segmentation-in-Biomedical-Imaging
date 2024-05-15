import torch
from torch import nn as nn

class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, dynamic_radius=None, mode=None, use_disks=False, overwrite_maps=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        self.dynamic_radius = dynamic_radius
        self.overwrite_maps = overwrite_maps
        self.mode = mode
        
        if self.mode not in ['disk', 'gaussian', 'distance']:
            raise Exception(f'Invalid mode: {self.mode}')
        
        self._warn_deperecation()
    
    def get_coord_features(self, points, batchsize, rows, cols):
        
        if self.mode == 'distance':
            # use legacy distance map implementation
            return self._get_distance_maps(points, batchsize, rows, cols)
        
        half_num_points = points.shape[1] // 2
        maps = torch.zeros((batchsize, 2, rows, cols), device=points.device)

        x_grid, y_grid = torch.meshgrid(torch.arange(rows, device=points.device), 
                                        torch.arange(cols, device=points.device), indexing='ij')

        for batch_idx in range(batchsize):
            for map_type in range(2):  # 0 for positive clicks, 1 for negative clicks
                for point_idx in range(half_num_points):
                    point = points[batch_idx, point_idx + (half_num_points * map_type)]
                    x, y, _, radius = point
                    if x < 0 or y < 0:  # Skip invalid points
                        continue

                    if self.mode == 'disk':
                        maps = self._draw_binary(x, y, radius, rows, cols, x_grid, y_grid, batch_idx, map_type, maps)
                    elif self.mode == 'gaussian':
                        maps = self._draw_gaussian(x, y, radius, x_grid, y_grid, batch_idx, map_type, maps)

        return maps.float()
    
    def _draw_binary(self, x, y, radius, rows, cols, x_grid, y_grid, batch_idx, map_type, maps):
        int_radius = int(torch.ceil(radius))
        x_min, x_max = max(0, int(x) - int_radius), min(rows, int(x) + int_radius + 1)
        y_min, y_max = max(0, int(y) - int_radius), min(cols, int(y) + int_radius + 1)
        mask = (x_grid[x_min:x_max, y_min:y_max] - x)**2 + (y_grid[x_min:x_max, y_min:y_max] - y)**2 <= radius**2
        maps[batch_idx, map_type, x_min:x_max, y_min:y_max][mask] = 1
        
        if self.overwrite_maps:
            maps[batch_idx, int(not map_type), x_min:x_max, y_min:y_max][mask] = 0
        
        return maps
    
    def _draw_gaussian(self, x, y, radius, x_grid, y_grid, batch_idx, map_type, maps):
        dx = x_grid - x
        dy = y_grid - y
        squared_distances = dx**2 + dy**2
        sigma = radius
        gaussian_map = torch.exp(-squared_distances / (sigma**2))
        maps[batch_idx, map_type, :, :] = torch.maximum(maps[batch_idx, map_type, :, :], gaussian_map)
        
        if self.overwrite_maps:
            maps[batch_idx, int(not map_type), :, :] = torch.minimum(maps[batch_idx, int(not map_type), :, :], 1 - gaussian_map)
        
        return maps
    
    def _get_distance_maps(self, points, batchsize, rows, cols):
        
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
        coords.div_(self.norm_radius * self.spatial_scale)
        coords.mul_(coords)
        
        coords[:, 0] += coords[:, 1]
        coords = coords[:, :1]
        coords[invalid_points, :, :, :] = 1e6
        
        coords = coords.view(-1, num_points, 1, rows, cols)
        coords = coords.min(dim=1)[0]
        coords = coords.view(-1, 2, rows, cols)
        coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])
    
    def _warn_deperecation(self):
        
        if self.use_disks:
            print(f'Property use_disks deprecated for DistMaps. Use mode=\'disk\' instead. Current mode: {self.mode}')
            
        if self.mode == 'distance':
            print(f'Distance map mode is bound to norm_radius. Current norm_radius: {self.norm_radius}')
        
        if self.cpu_mode:
            raise Exception('CPU mode is not supported')
            # from isegm.utils.cython import get_dist_maps
            # self._get_dist_maps = get_dist_maps
        
        if self.dynamic_radius is not None:
            print(f'{self.__class__} property dynamic_radius is deprecated. Click radius is decided by the clicker settings')
    