import numpy as np
from PIL import Image
from capstone_lab.campaign.data import ReadOnlySegDataset
from capstone_lab.errors import ArtifactError

class BackgroundDataset(ReadOnlySegDataset):
    @staticmethod
    def collate_fn(batch):
        item=ReadOnlySegDataset.collate_fn(batch)
        if item['masks'].shape[0]==0:
            for key in ['boundaries','distance_maps']:
                item[key]=item['masks'].clone()
        elif any(k not in item or item[k].shape!=item['masks'].shape for k in ['boundaries','distance_maps']):
            raise ArtifactError('Positive boundary supervision missing/misaligned')
        return item

    def get_labels(self):
        original=self.records
        try:
            self.records=[r for r in original if r.get('kind')!='background_only']
            positives=super().get_labels() if self.records else []
        finally:self.records=original
        positive={r['im_file']:r for r in positives};labels=[];self.label_files=[]
        for row in original:
            image=(self.root/row['image']).resolve();label=(self.root/row['label']).resolve()
            self.label_files.append(str(label))
            if row.get('kind')!='background_only':labels.append(positive[str(image)]);continue
            if not image.is_relative_to(self.root/'artifacts/s8_full_backgrounds/v1/images') or label!=self.root/'artifacts/s9_background/run_v1/labels/empty.txt' or label.read_text().strip():
                raise ArtifactError('Unapproved background source/label')
            with Image.open(image) as im:w,h=im.size;im.verify()
            labels.append(dict(im_file=str(image),shape=(h,w),cls=np.zeros((0,1),np.float32),bboxes=np.zeros((0,4),np.float32),segments=[],keypoints=None,normalized=True,bbox_format='xywh'))
        return labels
