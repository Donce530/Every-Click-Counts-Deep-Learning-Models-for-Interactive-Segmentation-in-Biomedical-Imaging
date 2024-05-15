import torch
from torchvision import transforms

def get_predictor(
    net,
    brs_mode,
    device,
):
    if brs_mode == 'UnetPlusPlus':
        predictor = UnetPlusPlusPredictor(
            net,
            device
        )

    elif brs_mode == 'UnetPlusPlusExplainable':
        predictor = UnetPlusPlusPredictorExplainable(
            net,
            device
        )
        
    else:
        raise NotImplementedError

    return predictor


class UnetPlusPlusPredictor(object):
    def __init__(
        self,
        model,
        device,
        net_clicks_limit=None,
        **kwargs,
    ):
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        self.net = model
        self.to_tensor = transforms.ToTensor()


    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = (
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()
        )

    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, "with_prev_mask") and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        # Backwards compatibility
        image_nd = input_image
        clicks_lists = [clicks_list]
        is_image_changed = False

        pred_logits = self._get_prediction(
            image_nd, clicks_lists, is_image_changed
        )

        prediction = torch.sigmoid(pred_logits)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        output = self.net(image_nd, points_nd)
        return output["instances"]

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[: self.net_clicks_limit]
            pos_clicks = [
                click.coords_and_indx for click in clicks_list if click.is_positive
            ]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [
                (-1, -1, -1)
            ]

            neg_clicks = [
                click.coords_and_indx for click in clicks_list if not click.is_positive
            ]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [
                (-1, -1, -1)
            ]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            "transform_states": None,
            "prev_prediction": self.prev_prediction.clone(),
        }

    def set_states(self, states):
        self.prev_prediction = states["prev_prediction"]


class UnetPlusPlusPredictorExplainable(object):
    def __init__(
        self,
        model,
        device,
        net_clicks_limit=None,
        **kwargs,
    ):
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        self.net = model
        self.to_tensor = transforms.ToTensor()


    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = (
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()
        )

    def get_prediction(self, clicker, prev_mask=None):
        info = {}

        clicks_list = clicker.get_clicks()

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, "with_prev_mask") and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        info['original_image'] = self.original_image.cpu().numpy()[0]
        info['prev_mask'] = prev_mask.cpu().numpy()[0, 0]
        info['input_image'] = input_image.cpu().numpy()[0]

        # Backwards compatibility
        image_nd = input_image
        clicks_lists = [clicks_list]
        is_image_changed = False
        
        info['image_transforms'] = {
            'image_nd': image_nd,
            'clicks_lists': clicks_lists,
            'is_image_changed': is_image_changed,
        }

        pred_logits, seg_info = self._get_prediction(
            image_nd, clicks_lists, is_image_changed
        )

        info['coarse_seg'] = seg_info
        info['coarse_seg_prediction'] = pred_logits.cpu().numpy()[0, 0]

        prediction = torch.sigmoid(pred_logits)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0], info

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        output, info = self.net(image_nd, points_nd)
        return output["instances"], info

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[: self.net_clicks_limit]
            pos_clicks = [
                click.coords_and_indx for click in clicks_list if click.is_positive
            ]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [
                (-1, -1, -1)
            ]

            neg_clicks = [
                click.coords_and_indx for click in clicks_list if not click.is_positive
            ]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [
                (-1, -1, -1)
            ]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            "transform_states": None,
            "prev_prediction": self.prev_prediction.clone(),
        }

    def set_states(self, states):
        self.prev_prediction = states["prev_prediction"]
