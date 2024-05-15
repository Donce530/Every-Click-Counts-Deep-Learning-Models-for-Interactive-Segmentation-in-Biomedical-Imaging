import numpy as np
from copy import deepcopy
import cv2

import matplotlib.pyplot as plt


class Clicker(object):
    def __init__(
        self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0
    ):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(
            np.logical_and(self.gt_mask, np.logical_not(pred_mask)),
            self.not_ignore_mask,
        )
        fp_mask = np.logical_and(
            np.logical_and(np.logical_not(self.gt_mask), pred_mask),
            self.not_ignore_mask,
        )

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class DynamicClicker(object):
    def __init__(
        self,
        gt_mask=None,
        init_clicks=None,
        ignore_label=-1,
        click_indx_offset=0,
        allow_overlap=True,
        mode="locked",
        size_range_modifier=0,
    ):
        self.click_indx_offset = click_indx_offset
        self.allow_overlap = allow_overlap
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

        self.mode = mode
        self.size_range_modifier = size_range_modifier

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        # Combine logical operations for efficiency
        fn_mask = self.gt_mask & ~pred_mask & self.not_ignore_mask
        fp_mask = ~self.gt_mask & pred_mask & self.not_ignore_mask

        # Apply padding if necessary
        if padding:
            pad_width = ((1, 1), (1, 1))
            fn_mask = np.pad(fn_mask, pad_width, "constant")
            fp_mask = np.pad(fp_mask, pad_width, "constant")

        # Calculate distance transforms
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        # Adjust distance transform if padding was applied
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]
            fn_mask = fn_mask[1:-1, 1:-1]
            fp_mask = fp_mask[1:-1, 1:-1]

        # Apply not clicked map
        fn_mask_dt *= self.not_clicked_map
        fp_mask_dt *= self.not_clicked_map

        # Determine if next click is positive and calculate min_disk_radius
        fn_max_dist = fn_mask_dt.max()
        fp_max_dist = fp_mask_dt.max()
        is_positive = fn_max_dist > fp_max_dist
        target_mask = fn_mask if is_positive else fp_mask

        # If there's nothing to click left to click
        if not np.any(target_mask):
            return Click(is_positive=True, coords=(-1, -1))

        target_mask_dt = fn_mask_dt if is_positive else fp_mask_dt
        coords_y, coords_x = np.unravel_index(
            target_mask_dt.argmax(), target_mask_dt.shape
        )

        if self.mode == "locked":
            radius = 5
        else:
            min_disk_radius = target_mask_dt[coords_y, coords_x]
            max_disk_radius = self._calculate_max_disk_radius(
                target_mask, (coords_y, coords_x)
            )
            radius = self._determine_radius(
                min_disk_radius, max_disk_radius, is_positive
            )

        return Click(
            is_positive=is_positive, coords=(coords_y, coords_x), disk_radius=radius
        )

    def _calculate_max_disk_radius(self, mask, click_coords, background_value=0):
        # Convert coordinates from (y, x) to (x, y)
        click_x, click_y = click_coords[1], click_coords[0]

        # Ensure coordinates are within bounds and not background
        if not (0 <= click_x < mask.shape[1] and 0 <= click_y < mask.shape[0]):
            return 0
        if mask[click_y, click_x] == background_value:
            return 0

        # Initialize a mask to record the visited pixels
        visited = np.zeros_like(mask, dtype=bool)

        # Start flood-fill from the click coordinates
        component_mask = self._flood_fill(
            mask, (click_x, click_y), visited, background_value
        )

        # Calculate minimum enclosing circle radius for the component
        points = np.column_stack(np.where(component_mask))
        center, radius = cv2.minEnclosingCircle(points)

        # Plotting
        # self._plot_results(mask, component_mask, center, radius, (click_x, click_y))

        return radius

    def _flood_fill(self, mask, start_coords, visited, background_value):
        stack = [start_coords]
        component_mask = np.zeros_like(mask, dtype=bool)

        while stack:
            x, y = stack.pop()
            if not visited[y, x] and mask[y, x] != background_value:
                visited[y, x] = True
                component_mask[y, x] = True

                for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                    if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]:
                        stack.append((nx, ny))

        return component_mask

    def _determine_radius(self, min_radius, max_radius, is_positive):
        # Determine base radius based on mode
        if self.mode == "locked":
            return 5
        if self.mode == "min":
            base_radius = min_radius
        elif self.mode == "max":
            base_radius = max_radius
        elif self.mode == "avg":
            base_radius = (min_radius + max_radius) / 2
        elif self.mode == "avg_only_pos":
            if not is_positive:
                return 5
            else:
                base_radius = (min_radius + max_radius) / 2
        elif self.mode == "distributed":
            # For 'distributed', we consider the full range, so set base_radius to None
            base_radius = None
        elif self.mode == "distributed_only_pos":
            if not is_positive:
                return 5
            else:
                base_radius = None
        else:
            raise Exception(f"Clicker mode {self.mode} unknown")

        # Calculate and return the radius if size_range_modifier is applied
        if self.size_range_modifier != 0 or self.mode.startswith("distributed"):
            if self.mode.startswith("distributed"):
                min_value = min_radius * (1 - self.size_range_modifier)
                max_value = max_radius * (1 + self.size_range_modifier)
            else:
                min_value = base_radius * (1 - self.size_range_modifier)
                max_value = base_radius * (1 + self.size_range_modifier)
            return np.random.uniform(min_value, max_value)

        # If size_range_modifier is not applied, return the base_radius
        # For 'distributed' mode, this part is not reached as base_radius is None
        return base_radius

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            if self.allow_overlap:
                self.not_clicked_map[coords[0], coords[1]] = False
            else:
                self._update_not_clicked_map_for_disk(
                    click.coords, click.disk_radius, clicked=True
                )

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            if self.allow_overlap:
                self.not_clicked_map[coords[0], coords[1]] = True
            else:
                self._update_not_clicked_map_for_disk(
                    click.coords, click.disk_radius, clicked=False
                )

    def _update_not_clicked_map_for_disk(self, center, radius, clicked):
        int_radius = int(np.ceil(radius))
        radius_squared = int_radius**2

        # Create a grid of coordinates
        x = np.arange(self.gt_mask.shape[0])
        y = np.arange(self.gt_mask.shape[1])
        xx, yy = np.meshgrid(x, y, indexing="ij")

        # Compute a mask for the disk
        disk_mask = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius_squared

        # Update the not_clicked_map using the disk mask
        self.not_clicked_map[disk_mask] = not clicked

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None, disk_radius=5):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx
        self.disk_radius = disk_radius

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    @property
    def as_tuple(self):
        return (*self.coords, self.indx, self.disk_radius)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy

    @property
    def description(self):
        return f"{self.coords}, {self.is_positive}, {self.disk_radius}"
