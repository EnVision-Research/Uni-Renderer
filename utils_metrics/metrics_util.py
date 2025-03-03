import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_miou_per_batch(pred_mask, gt_mask):
    # Flatten the masks to simplify calculations
    pred_mask_flat = pred_mask.view(-1)
    gt_mask_flat = gt_mask.view(-1)
    
    # Identify all unique classes in both masks
    unique_classes = torch.unique(torch.cat((pred_mask_flat, gt_mask_flat)))
    
    # Initialize the sum of IoUs and the count of encountered classes
    iou_sum = 0.0
    encountered_classes = 0
    
    # Calculate IoU for each class
    for cls in unique_classes:
        # Find the pixels that belong to the current class for both masks
        pred_cls = pred_mask_flat == cls
        gt_cls = gt_mask_flat == cls
        
        # Calculate the intersection and union
        intersection = (pred_cls & gt_cls).sum().item()
        union = (pred_cls | gt_cls).sum().item() - intersection
        
        # Avoid division by zero
        if union == 0:
            continue
        
        # Update the IoU sum and the count of encountered classes
        iou_sum += intersection / union
        encountered_classes += 1
    
    # Calculate mean IoU
    miou = iou_sum / encountered_classes if encountered_classes > 0 else 0
    
    return miou



class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task. 

    Attributes:
        record (list): A list of the metric scores for each sample.
    """
    def __init__(self):
        self.record = []
    
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in a batch and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor for each batch.
            gt (torch.Tensor): The ground-truth tensor for each batch.
        """
        bs = pred.shape[0]
        for i in range(bs):
            self.compute_metric(pred[i], gt[i])

    @property
    def compute_metric(self, pred, gt):
        r"""Calculate the metric scores for each sample.

        Args:
            pred (torch.Tensor): The prediction tensor for each sample.
            gt (torch.Tensor): The ground-truth tensor for each sample.
        """
        pass
    
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        return [np.mean(np.array(self.record))]
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []


# seg
class SegMetric(AbsMetric):
    # https://github.com/pytorch/vision/blob/main/references/segmentation/utils.py#L66
    def __init__(self, num_classes):
        super(SegMetric, self).__init__()
        
        self.num_classes = num_classes
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        
    def compute_metric(self, pred, gt):
        self.record = self.record.to(pred.device)
        # pred = pred[:self.num_classes,:,:].softmax(0).argmax(0).flatten()
        pred = pred.flatten()
        gt = gt.long().flatten()
        k = (gt >= 0) & (gt < self.num_classes)
        inds = self.num_classes * gt[k].to(torch.int64) + pred[k]
        self.record += torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        
    def score_fun(self):
        h = self.record.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h).sum() / h.sum()
        return [torch.mean(iu).item(), acc.item()]
    
    def reinit(self):
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

# depth
class DepthMetric(AbsMetric):
    def __init__(self):
        super(DepthMetric, self).__init__()
        
    def compute_metric(self, pred, gt):
        device = pred.device
        binary_mask = (torch.sum(gt, dim=0) != 0).unsqueeze(0).to(device)
        pred = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        abs_err = torch.abs(pred - gt)
        rel_err = torch.abs(pred - gt) / gt
        abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        self.record.append([abs_err, rel_err])
        
    def score_fun(self):
        results = tuple(zip(*self.record))
        assert len(results) == 2

        return [np.mean(np.array(results[i])) for i in range(2)]
    
# normal
class NormalMetric(AbsMetric):
    def __init__(self):
        super(NormalMetric, self).__init__()
        
    def compute_metric(self, pred, gt):
        binary_mask = (torch.sum(gt, dim=0) != 0) # [h, w]
        gt = gt / torch.norm(gt, p=2, dim=0, keepdim=True) # [3, h, w]
        pred = pred / torch.norm(pred, p=2, dim=0, keepdim=True) # [3, h, w]
        error = torch.acos(torch.clamp(torch.sum(pred*gt, 0).masked_select(binary_mask), -1, 1)).detach().cpu().numpy() # 1-D vector
        error = np.degrees(error)
        self.record.append([np.mean(error), np.median(error), \
                           np.mean((error < 11.25)*1.0), np.mean((error < 22.5)*1.0), \
                           np.mean((error < 30)*1.0)])

    def score_fun(self):
        results = tuple(zip(*self.record))
        #print(results)
        assert len(results) == 5

        return [np.mean(np.array(results[i])) for i in range(5)]
    