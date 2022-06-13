import os
import os.path
import json
from typing import Any, Callable, cast, Optional, Tuple
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

class MET_queries(VisionDataset):

    def __init__(
            self,
            root: str = ".",
            test: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            im_dir = None
    ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        if test:
            fn = "testset.json"
        else:
            fn = "valset.json"

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
        
            samples.append(e['path'])
            if "MET_id" in e:
                targets.append(int(e['MET_id']))
            else:
                targets.append(-1)

        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.im_dir = im_dir

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.im_dir is not None:
            path = os.path.join(self.im_dir, "images/" + self.samples[index])            

        else:
            path = os.path.join(os.path.dirname(self.root), "images/" + self.samples[index])
        
        target = self.targets[index]

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        
        return len(self.samples)


class MET_database(VisionDataset):

    def __init__(
            self,
            root: str = ".",
            mini: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            im_dir = None
    ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        fn = "MET_database.json"

        if mini:
            fn = "mini_"+fn

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
            samples.append(e['path'])
            targets.append(int(e['id']))

        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.im_dir = im_dir


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        if self.im_dir is not None:
            path = os.path.join(self.im_dir, "images/" + self.samples[index])            

        else:
            path = os.path.join(os.path.dirname(self.root), "images/" + self.samples[index])

        target = self.targets[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        
        return len(self.samples)
