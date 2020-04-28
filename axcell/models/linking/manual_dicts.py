#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

metrics = {
    'Accuracy': ['acc', 'accuracy'],
    'BLEU': ['bleu'],
    'BLEU score': ['bleu'],
    'Character Error Rate': ['cer', 'cers'],
    'Error': ['error', 'err', 'error rate'],
    'Exact Match Ratio': ['exact match'],
    'F1': ['f1', 'f1 score'],
    'F1 score': ['f1', 'f1 score'],
    'MAP': ['map'],
    'Percentage error': ['wer', 'per', 'wers', 'pers', 'word error rate', 'word error rates', 'phoneme error rates',
                         'phoneme error rate', 'error', 'error rate', 'error rates'],
    'Word Error Rate': ['wer', 'wers', 'word error rate', 'word error rates', 'error', 'error rate', 'error rates'],
    'Word Error Rate (WER)': ['wer', 'wers', 'word error rate', 'word error rates', 'error', 'error rate', 'error rates'],
    'Word Accuracy': ['accuracy', 'word accuracy', 'acc', 'word acc'],
    'ROUGE-1': ['r1'],
    'ROUGE-2': ['r2'],
    'ROUGE-L': ['rl'],
    'Precision': ['precision'],
    'Recall': ['recall'],
    # RAIN REMOVAL
    'PSNR': ['psnr', 'psnr (db)', 'mean psnr'],
    'SSIM': ['ssim'],
    'UQI': ['uqi'],
    'VIF': ['vif'],
    'SSEQ': ['sseq'],
    'NIQE': ['niqe'],
    'BLINDS-II': ['blinds-ii'],
    'FSIM': ['fsim'],
    # SEMANTIC SEGMENTATION
    'Mean IoU': ['miou', 'mean iou', 'mean iu', 'class iou', 'iou cla', 'cla iou'],
    'Pixel Accuracy': ['pixel accuracy', 'pixel acc', 'pixel acc.', 'pixacc', 'pixel'],
    'Category IoU': ['cat iou', 'iou cat'],
    'class iIoU': ['class iiou', 'iiou cla'],
    'Category iIoU': ['cat iiou', 'iiou cat'],
    'Mean Accuracy': ['mean acc', 'mean', 'acc', 'accuracy', 'mean accuracy'],
    'Mean Error': ['mean err', 'mean', 'err', 'mean error', 'error'],
    'Top-1 Accuracy': ['top 1 accuracy', 'top 1', 'top 1 acc'],
    'Top-5 Accuracy': ['top 5 accuracy', 'top 5', 'top 5 acc'],
    'Top-1 Error Rate': ['top 1 error', 'top 1', 'top 1 err'],
    'Top-5 Error': ['top 5 error', 'top 5', 'top 5 err']
}

# datasets[taxonomy name] is a list of normalized evidences for taxonomy name
datasets = {
    'Hub5\'00 Average': ['avg', 'full', 'hub5', 'sum', 'evaluation'],
    'Hub5\'00 Switchboard': ['swbd', 'swb', 'hub5 swb', 'hub5 swbd', 'switchboard'],
    'Hub5\'00 CallHome': ['ch', 'hub5 ch', 'call home', 'chm'],
    'TIMIT': ['timit'],
    'WSJ eval92': ['wsj eval 92', 'eval 92', 'wsj'],
    'WSJ eval93': ['wsj eval 93', 'eval 93', 'wsj'],
    'LibriSpeech test-clean': ['libri speech test clean', 'libri speech', 'test', 'tst', 'clean', 'test clean'],
    'LibriSpeech test-other': ['libri speech test other', 'libri speech', 'test', 'tst', 'other', 'test other',
                               'noisy'],
    'Babel Cebuano': ['babel cebuano', 'babel', 'cebuano', 'ceb'],
    'Babel Kazakh': ['babel kazakh', 'babel', 'kazakh', 'kaz'],
    'Babel Kurmanji': ['babel kurmanji', 'babel', 'kurmanji', 'kur'],
    'Babel Lithuanian': ['babel lithuanian', 'babel', 'lithuanian', 'lit'],
    'Babel Telugu': ['babel telugu', 'babel', 'telugu', 'tel'],
    'Babel Tok Pisin': ['babel tok pisin', 'babel', 'tok pisin', 'tok'],

    'Ask Ubuntu': ['ask ubuntu', 'ask u', 'ubuntu'],
    'Chatbot': ['chatbot'],
    'Web Apps': ['web apps'],
    'CHiME clean': ['chime clean', 'chime', 'clean'],
    'CHiME real': ['chime real', 'chime', 'real'],
    'CHiME simu': ['chime simu', 'chime', 'simu', 'sim', 'simulated'],
    'CHiME-4 real 6ch': ['chime 4 real 6 ch', 'chime 4', 'real', '6 channel'],
    'AG News': ['ag news', 'ag'],
    'GigaWord': ['gigaword', 'giga'],
    'GEOTEXT': ['geotext', 'geo'],
    'IWSLT2015 English-Vietnamese': ["iwslt 2015 english vietnamese", "iwslt", "2015", "english vietnamese", "en vi",
                                      "iwslt 15 english vietnamese", "iwslt 15 en vi", "english", "en", "vietnamese",
                                      "vi"],
    'IWSLT2011 English TED Talks': ["iwslt 2011 english ted talks", "iwslt", "2011", "english", "en", "eng", "ted",
                                    "ted talks", "english ted talks"],
    'IWSLT2012 English TED Talks': ["iwslt 2012 english ted talks", "iwslt", "2012", "english", "en", "eng", "ted",
                                    "ted talks", "english ted talks"],
    'IWSLT2014 English-German': ["iwslt 2014 english german", "iwslt", "2014", "english german", "en de", "en", "de",
                                 "english", "german"],
    'Rich Transcription 2002': ["rich transcription 2002", "rich transcription 02", "rt 2002", "2002", "rt 02", "rich",
                                "transcription"],
    'Rich Transcription 2003': ["richt ranscription 2003", "rich transcription 03", "rt 2003", "2003", "rt 03", "rich",
                                "transcription"],
    'Rich Transcription 2004': ["rich transcription 2004", "rich transcription 04", "rt 2004", "2004", "rt 04", "rich",
                                "transcription"],
    'DIRHA English WSJ real': ['dirha english wsj real', 'dirha', 'english', 'en', 'eng', 'real', 'wsj'],
    'DIRHA English WSJ simu': ['dirha english wsj simu', 'dirha', 'english', 'en', 'eng', 'simu', 'wsj', 'simulated'],
    'VCTK clean': ["vctk clean", "vctk", "clean", "voice bank", "voice", "bank", "corpus"],
    'VCTK noisy': ["vctk noisy", "vctk", "noisy", "voice bank", "voice", "bank", "corpus"],
    'VoxForge American-Canadian': ["vox forge american canadian", "vox forge", "vox", "forge", "american canadian",
                                   "american", "canadian", "us ca"],
    'VoxForge Commonwealth': ["vox forge common wealth", "vox forge", "common wealth", "vox", "forge", "common",
                              "wealth"],
    'VoxForge European': ["vox forge european", "vox forge", "european", "vox", "forge", "eu"],
    'VoxForge Indian': ["vox forge indian", "vox forge", "indian", "vox", "forge"],

    # Face Alignment
    'AFLW test': ['annotated faces in the wild', 'annotated facial landmarks in the wild', 'facial', 'landmark',
                  'annotated', 'faces', 'in the wild', 'faces in the wild'],

    # Human Part Segmentation
    'CIHP val': ['crowd instance level human parsing'],
    'MHP v2.0 val': ['multi human parsing'],

    # Image Generation
    # 'LSUN Bedroom': ['large scale scene understanding'],

    # RAIN REMOVAL
    'Raindrop': ['raindrop'],
    'Rain100H': ['rain100h'],
    'Rain100L': ['rain100l'],
    'Rain12': ['rain12'],
    'Rain800': ['rain800'],
    'Rain1400': ['rain1400'],
    'Real Rain': ['real rain'],
    'Rain in Surveillance': ['ris'],
    'Rain in Driving': ['rid'],
    'DID-MDN': ['did-mdn'],
    'SOTS': ['sots'],
    'Test 1': ['test 1'],
    'RainSynLight25': ['rainsynlight25'],
    'RainSynComplex25': ['rainsyncomplex25'],
    'NTURain': ['nturain'],
    'RainSynAll100': ['rainsynall100'],
    'SPA-DATA': ['spa-data'],
    'LasVR': ['lasvar'],
    # SEMANTIC SEGMENTATION
    'ADE20K': ['ade20k'],
    'ADE20K test': ['ade20k test', 'ade20k', 'test'],

    'COCO-Stuff': ['coco stuff', 'coco', 'stuff'],
    'PASCAL VOC 2012': ['voc 12', 'pascal voc 12', 'voc12' 'pascal'],
    'PASCAL VOC 2012 test': ['voc 12', 'pascal voc 12', 'voc12' 'pascal'],
    'PASCAL VOC 2011': ['voc 11', 'pascal voc 11', 'voc11' 'pascal'],
    'PASCAL VOC 2011 test': ['voc 11', 'pascal voc 11', 'voc11', 'pascal'],

    'ImageNet': ['imagenet'],
    'Cityscapes test': ['cityscapes'],
    'PASCAL Context': ['pascal context', 'pascal', 'context', 'pascalcontext', 'pascal-context'],
    'PASCAL Context val': ['pascal context', 'pascal', 'context', 'pascalcontext', 'val', 'pascal-context'],
    'PASCAL Person-Part': ['pascal person part'],
    'ParseNet': ['parsenet'],
    'LIP': ['lip'],
    'SUN-RGBD': ['sun rgbd', 'sunrgbd', 'sunrgb d'],
    'NYU Depth v2': ['nyudv2'],

    # Gender Classification
    'LFWA': ['labeled faces in the wild', 'faces in the wild', 'faces', 'in the wild'],

    # Recommendation Systems
    'NeurIPS Co-authorship': ['nips'],

    # Sentiment Analysis
    'Pang and Lee 2004': ['pl04'],

    # Semantic Textual Similarity
    'STS Benchmark': ['sts b'],

    # Paraphrase Identification
    'Microsoft Research Paraphrase Corpus': ['MRPC'],
}

tasks = {}

complementary_metrics = {k.lower(): v for k, v in {
    'Accuracy': 'Error',
    'Error': 'Accuracy',
    'Acc': 'Err',
    'Err': 'Acc',
    'Percentage Error': 'Accuracy',
    'Error rate': 'Accuracy',
    'Word Error Rate': 'Word Accuracy',
    'Word Error Rate (WER)': 'Word Accuracy',
    'Top-1 Accuracy': 'Top-1 Error Rate',
    'Top-3 Accuracy': 'Top-3 Error Rate',
    'Top-5 Accuracy': 'Top-5 Error Rate',
    'Top 1 Accuracy': 'Top 1 Error Rate',
    'Top 3 Accuracy': 'Top 3 Error Rate',
    'Top 5 Accuracy': 'Top 5 Error Rate',
    'Top-1 Error Rate': 'Top-1 Accuracy',
    'Top-3 Error Rate': 'Top-3 Accuracy',
    'Top-5 Error Rate': 'Top-5 Accuracy',
    'Top 1 Error Rate': 'Top 1 Accuracy',
    'Top 3 Error Rate': 'Top 3 Accuracy',
    'Top 5 Error Rate': 'Top 5 Accuracy',
    'Top-1 Error': 'Top-1 Accuracy',
    'Top-3 Error': 'Top-3 Accuracy',
    'Top-5 Error': 'Top-5 Accuracy',
    'Top 1 Error': 'Top 1 Accuracy',
    'Top 3 Error': 'Top 3 Accuracy',
    'Top 5 Error': 'Top 5 Accuracy',
    'Classification Accuracy': 'Classification Error',
    'Classification Error': 'Classification Accuracy',
}.items()}

stop_words = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with"
}
