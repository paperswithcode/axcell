#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re

dataset_name_re = re.compile(r"\b(the)\b\s*(?P<name>((?!(the)\b)\w+\W+){1,10}?)(test|val(\.|idation)?|dev(\.|elopment)?|train(\.|ing)?\s+)?\bdata\s*set\b", re.IGNORECASE)

parens_re = re.compile(r"\([^)]*?\)|\[[^]]*?\]")
def remove_parens(text):
    return parens_re.sub("", text)

def clean_name(name):
    return remove_parens(name.strip()).strip()

year_2k_re = re.compile(r"20(\d\d)")
hyphens_re = re.compile(r"[-_'`–’→]")
ws_re = re.compile(r"\s+")
dataset_prefix_re = re.compile(r"[A-Z]|[a-z]+[A-Z]+|[0-9]")

def normalize_dataset(name):
    name = hyphens_re.sub(" ", name)
    name = year_2k_re.sub(r"\1", name)
    name = ws_re.sub(" ", name)
    return name.strip().lower()

## temporarily moved to notebook
# class DatasetExtractor:
#     def from_paper(self, paper):
#         text = paper.text.abstract
#         if hasattr(paper.text, "fragments"):
#             text += " ".join(f.text for f in paper.text.fragments)
#         return self(text)
#
#     def __call__(self, text):
#         extracted = [clean_name(m.group("name")) for m in dataset_name_re.finditer(text)]
#         print("Extracted:", extracted)
#         cleaned = [x for x in extracted if dataset_prefix_re.match(x)]
#         print("Cleaned:", cleaned)
#         return cleaned
#         filtered = list(set([x for x in cleaned if normalize_dataset(x) in normalized_datasets]))
#         print("Filtered:", filtered)
#         return filtered


datasets = ['VOT2016', 'Penn Treebank', 'DIV2K', 'SCUT-FBP5500', 'SCUT-FBP', 'ImageNet', 'KITTI', 'Cityscapes', 'Street View House Number', 'MNIST', '1000-class ImageNet', 'CIFAR-10', 'Berkeley Segmentation', 'AFLW', 'BIWI', '300W-LP', 'AFLW2000', 'AFW', 'Stanford Question Answering', 'SQuAD', '80 million tiny images', 'PASCAL VOC 2012', 'ILSVRC-2012 ImageNet', 'CIFAR-100', 'NewsQA', 'COCO', 'Market-1501', 'LSUN', 'Matterport3D', 'Market1501', 'bAbI', 'WikiHop', 'MICC', 'Wild', 'Yelp', 'SNLI', 'MultiNLI', 'Age', 'Yahoo', 'OMNIGLOT', 'DSTC2', 'Cars', 'CBT', 'CNN', 'Daily Mail', 'Jester', 'Adult', 'LSUN bedroom', 'CUB', 'Caltech-UCSD Birds-200-2011', 'Street View House Numbers', 'TREC QA', 'Realtor360', 'PanoContext', 'Stanford 2D-3D', 'Camelyon16', 'COCO-Stuff', 'Flickr Landscapes', 'ADE20K', 'MSRA', 'OntoNotes', 'Visual Question Answering', 'VQA', 'VQA v2.0', 'Indian Pines', 'Pavia University', 'MR', 'PASCAL3D+', 'PASCAL VOC 2007', 'VOC 2007', 'LSP', 'VIPeR', 'PASCAL VOC', 'ImageNet detection', 'MS-COCO', 'Caltech-UCSD Birds', 'MPII Human Pose', 'CoNLL 2003 NER', 'FCE', 'Cora', 'Wikipedia', 'Switchboard', '1B word', 'SVHN', 'Caltech pedestrian', 'Set5', 'Urban100', 'AVA', 'Charades', 'MMI', 'Extended Cohn-Kanade', 'CKP', 'ICDAR 2015', 'SwDA', 'MRDA', 'ModelNet', 'PASCAL 3D', 'ShapeNet', 'TriviaQA', 'Facescrub', 'NYUV2', 'ShapeNet part', 'WSJ', 'CoNLL03 NER', 'NER', 'CoNLL03', 'LibriSpeech', '300W', 'WN18', 'ILSVRC 2012 classification', 'Penn Tree Bank', 'Cifar-10', 'SQuAD 2.0', 'PTB', 'DukeMTMC-reID', 'CUHK03', 'SearchQA', 'Stanford Natural Language Inference', 'NYU', 'ICVL', 'NYU hand pose', 'WN18RR', 'CoNLL-2005 shared task', 'CoNLL-2012 shared task', 'CoNLL-2005', 'CoNLL-2012', 'ImageNet 2012', '300-W', 'AFLW2000-3D', 'LFW', 'Omniglot', 'PROMISE 2012', 'Twitter', 'Florence', 'SUN-RGBD', 'Microsoft COCO', 'ImageNet classification', 'Something-Something', 'MRC', 'MS MARCO', 'Amazon', 'Alibaba', 'Netflix', 'PASCAL-Person-Part', 'CIHP', 'Pascal VOC', 'MS-Celeb-1M', 'CASIA', 'MegaFace', 'IJB-B', 'ImageNet-1k', 'Places365-Standard', 'SciTail', 'GTSRB', 'GRID', 'BSD', 'LIVE1', 'CNN/Daily Mail', 'Caltech', 'MS COCO', 'Restaurant', 'JSB Chorales', 'CUHK', 'CUFSF', 'JFT-300M', 'CelebA', 'RaFD', 'Amazon Reviews', 'Amazon reviews', 'SemEval', 'Tobacco-3482', 'RVL-CDIP', 'Douban', 'Company\xe2\x88\x97', 'Criteo', 'Semantic Boundaries', 'Caltech-UCSD birds', 'IMDb', 'VGG-Face', 'MoFA', 'FERET', 'iNat2017', 'ScanNet', 'TIMIT', 'VOC 2012', 'SICK', 'IJB-A', 'CACD', 'MSCeleb', 'YTF', 'CACD-VS', 'CityScapes', 'COCO detection', 'Bosch', 'LISA', 'Tsinghua-Tencent', 'FDDB', 'Mikolajczyk', 'Middlebury', 'Kitti', 'ILSVRC2012', 'BSD100', 'LineMod', 'Occlusion', 'GTAV', 'CityPersons', 'ETH', 'INRIA', 'ILSVRC CLS-LOC', 'Caltech-USA', 'BlogCatalog', 'CoNLL', 'MPII', 'Cityscapes', 'Cityscapes', 'CamVid', 'Amazon Review', 'STL-10', 'Imagenet', 'ShapeNet-Part', 'ModelNet40', 'BUS 2017', 'Quora Question Pairs', 'SST', 'MARS', 'PRW', 'BSD68', 'IMDB', 'ASPEC', 'OTB-2015', 'VOT-2017 public', 'Tejani', 'LineMOD', 'CASIA WebFace', 'Flying Chairs', 'FLIC', 'Set14 \xc3\x974', 'Human3.6M', 'Google News', 'Jobs', 'WikiText-2', 'Rotten Tomatoes', 'RCV1', 'WIDER FACE val', 'WIDER FACE', 'COCO', 'PoseTrack', 'HPatches', 'MHP v2.0', 'Buffy', 'ShapeNetCore', 'EVAL', 'MAFA', 'iPinYou', 'CASIA-WebFace', 'JANUS CS2', 'Cross-City', 'GTA5', 'SYNTHIA', 'MovieLens-100k', 'MovieLens-1M', 'LAMBADA', 'bAbi', 'Visual Genome', 'Visual-7W', 'Google-Ref', 'CelebA-HQ', 'PASCAL', 'QASent', 'WikiQA', 'Online Products', 'FB15k-237', 'MovieLens 1M', 'REST', 'Yosemite', 'PASCAL faces', 'MusicNet', 'Multi-MNIST', 'CLEVR', 'Quora', 'Who Did What', 'Children\xe2\x80\x99s Book', 'Set14', 'CFP', 'CTW1500', 'Weizmann Horse', 'ReVerb45K', 'AG\xe2\x80\x99s News', 'WMT En\xe2\x86\x92Fr', 'WMT En\xe2\x86\x92De', 'CNN/DailyMail', 'NYT', 'ECCV HotOrNot', 'bAbI story-based QA', 'PPI', 'Mini-ImageNet', 'ITOP', 'YCB-Video', 'DFW', 'ACL-ARC', 'SciCite', 'HumanEva', 'LINEMOD', 'Occlusion LINEMOD', 'Face Detection', 'UP-3D', 'WT2', 'PASCAL-Context', 'TREC', 'WDW', 'Shoulder-Pain', 'MovieLens', 'CT-150', 'WMT', 'CMU-MOSI', 'IEMOCAP', 'MPII Multi-Person Pose', '91-image', 'CoNLL 2003', 'COCO keypoint detection', 'WiderFace', 'Extended Yale B', 'Hutter Prize', 'SST-1', 'CUB-200-2011', 'Cars196', 'Stanford Online Products', 'Caltech and KITTI', 'BRATS', 'E2E', 'TV', 'Laptop', 'CIFAR', 'CHALL_H80K', 'VQA v2', 'NYU depth', 'NYUD', 'Cityscape', 'IBUG', 'BP4D', 'CAF', 'LexNorm2015', 'YouTube Face', 'DAQUAR', 'NYUDv2', 'SmallTobacco', 'BigTobacco', 'TID2013', 'CK+', 'PubMed 20k', 'WAF', 'MPII Multi-Person', 'GTA', 'PCSO mugshot', 'CIFAR100', 'ImageNet', 'MHP', 'CompCars', 'CUB200-2011 bird', 'CUHK03 labeled', 'Stanford 2D-3D annotation', 'Reddit', 'Stanford SQuAD', 'Graph Reachability', 'AIDA-B', 'VGG face', 'Yahoo! Answer', 'AR', 'Caltech Pedestrian', 'CARS-196', 'Pascal Context', 'Scan2CAD', 'Tiny Images', 'CAT', 'CIFAR10', 'JFT', 'PA-100K', 'VOC2007', 'Wikihop', 'PASCAL face', 'MPQA', 'NELL995', 'NELL-995', 'ShanghaiTech', 'SARC', 'Pol', 'CUHK03 detected', 'Celeb-Seq', 'ICDAR2015 Incidental Scene Text', 'Stanford Sentiment Treebank', 'CoQA', 'Massachusetts roads', 'MPIIGaze', 'SBD', 'InsuranceQA', 'ETHZ', 'Landmarks', 'H36M', 'OccludedLINEMOD', 'UCF101', 'RGBD', 'USPS', 'Visual QA', 'COCO-QA', 'Vid4', 'DAVIS-10']
normalized_datasets = [normalize_dataset(ds) for ds in datasets]
