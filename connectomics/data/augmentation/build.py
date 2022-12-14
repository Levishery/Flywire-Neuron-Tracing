from yacs.config import CfgNode

from .composition import Compose
from .warp import Elastic
from .grayscale import Grayscale
from .flip import Flip
from .rotation import Rotate
from .rescale import Rescale
from .misalign import MisAlignment
from .missing_section import MissingSection
from .missing_parts import MissingParts
from .motion_blur import MotionBlur
from .cutblur import CutBlur
from .cutnoise import CutNoise
from .copy_paste import CopyPasteAugmentor

def build_train_augmentor(cfg: CfgNode, keep_uncropped: bool = False, keep_non_smoothed: bool = False, ssl_augmenter: bool = False):
    r"""Build the training augmentor based on the options specified in the configuration
    file. 

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        keep_uncropped (bool): keep uncropped data in the output. Default: `False`
        keep_non_smoothed (bool): keep the masks before smoothing in the output. Default: `False`
        ssl_augmentor (bool): build ssl augmentor in ssl pipeline, otherwise return empty augmentor with larger input. Default: `False`

    Note:
        The two arguments, keep_uncropped and keep_non_smoothed, are used only for debugging,
        which are `False` by defaults and can not be adjusted in the config file.
    """
    aug_list = []
    
    names = cfg.AUGMENTOR.ADDITIONAL_TARGETS_NAME
    types = cfg.AUGMENTOR.ADDITIONAL_TARGETS_TYPE
    if names is None:
        additional_targets = None
    else:
        assert len(names) == len(types)
        additional_targets = {}
        for i in range(len(names)):
            additional_targets[names[i]] = types[i]

    #1. rotate
    if cfg.AUGMENTOR.ROTATE.ENABLED:
        aug_list.append(
            Rotate(rot90=cfg.AUGMENTOR.ROTATE.ROT90,
                   p=cfg.AUGMENTOR.ROTATE.P,
                   additional_targets=additional_targets))

    #2. rescale
    if cfg.AUGMENTOR.RESCALE.ENABLED:
        aug_list.append(
            Rescale(p=cfg.AUGMENTOR.RESCALE.P,
                    additional_targets=additional_targets))

    #3. flip
    if cfg.AUGMENTOR.FLIP.ENABLED:
        aug_list.append(
            Flip(do_ztrans=cfg.AUGMENTOR.FLIP.DO_ZTRANS,
                 p=cfg.AUGMENTOR.FLIP.P,
                 additional_targets=additional_targets))

    #4. elastic
    if cfg.AUGMENTOR.ELASTIC.ENABLED:
        aug_list.append(
            Elastic(alpha=cfg.AUGMENTOR.ELASTIC.ALPHA,
                    sigma=cfg.AUGMENTOR.ELASTIC.SIGMA,
                    p=cfg.AUGMENTOR.ELASTIC.P,
                    additional_targets=additional_targets))

    #5. grayscale
    if cfg.AUGMENTOR.GRAYSCALE.ENABLED:
        aug_list.append(
            Grayscale(p=cfg.AUGMENTOR.GRAYSCALE.P,
                      additional_targets=additional_targets))

    #6. missingparts
    if cfg.AUGMENTOR.MISSINGPARTS.ENABLED:
        aug_list.append(
            MissingParts(iterations=cfg.AUGMENTOR.MISSINGPARTS.ITER,
                         p=cfg.AUGMENTOR.MISSINGPARTS.P,
                         additional_targets=additional_targets))

    if not (cfg.DATASET.CONNECTOR_DATSET and cfg.DATASET.RELABEL):
        # 7. missingsection, should be done after relabel
        if cfg.AUGMENTOR.MISSINGSECTION.ENABLED and not cfg.DATASET.DO_2D:
            aug_list.append(
                MissingSection(
                    num_sections=cfg.AUGMENTOR.MISSINGSECTION.NUM_SECTION,
                    p=cfg.AUGMENTOR.MISSINGSECTION.P,
                    additional_targets=additional_targets))
        #8. misalignment, should be done after relabel
        if cfg.AUGMENTOR.MISALIGNMENT.ENABLED and not cfg.DATASET.DO_2D:
                aug_list.append(
                    MisAlignment(
                        displacement=cfg.AUGMENTOR.MISALIGNMENT.DISPLACEMENT,
                        rotate_ratio=cfg.AUGMENTOR.MISALIGNMENT.ROTATE_RATIO,
                        p=cfg.AUGMENTOR.MISALIGNMENT.P,
                        additional_targets=additional_targets))

    #9. motion-blur
    if cfg.AUGMENTOR.MOTIONBLUR.ENABLED:
        aug_list.append(
            MotionBlur(
                sections=cfg.AUGMENTOR.MOTIONBLUR.SECTIONS,
                kernel_size=cfg.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE,
                p=cfg.AUGMENTOR.MOTIONBLUR.P,
                additional_targets=additional_targets))

    #10. cut-blur
    if cfg.AUGMENTOR.CUTBLUR.ENABLED:
        aug_list.append(
            CutBlur(length_ratio=cfg.AUGMENTOR.CUTBLUR.LENGTH_RATIO,
                    down_ratio_min=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN,
                    down_ratio_max=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX,
                    downsample_z=cfg.AUGMENTOR.CUTBLUR.DOWNSAMPLE_Z,
                    p=cfg.AUGMENTOR.CUTBLUR.P,
                    additional_targets=additional_targets))

    #11. cut-noise
    if cfg.AUGMENTOR.CUTNOISE.ENABLED:
        aug_list.append(
            CutNoise(length_ratio=cfg.AUGMENTOR.CUTNOISE.LENGTH_RATIO,
                     scale=cfg.AUGMENTOR.CUTNOISE.SCALE,
                     p=cfg.AUGMENTOR.CUTNOISE.P,
                     additional_targets=additional_targets))

    #12. copy-paste
    if cfg.AUGMENTOR.COPYPASTE.ENABLED:
        aug_list.append(
            CopyPasteAugmentor(aug_thres=cfg.AUGMENTOR.COPYPASTE.AUG_THRES,
                     p=cfg.AUGMENTOR.CUTNOISE.P,
                     additional_targets=additional_targets))

    # compose the list of transforms
    if not (cfg.DATASET.CONNECTOR_DATSET and cfg.DATASET.RELABEL):
        augmentor = Compose(transforms=aug_list,
                            input_size=cfg.MODEL.INPUT_SIZE,
                            smooth=cfg.AUGMENTOR.SMOOTH,
                            keep_uncropped=keep_uncropped,
                            keep_non_smoothed=keep_non_smoothed,
                            additional_targets=additional_targets)
    else:
        aug_list_after = []
        if cfg.AUGMENTOR.MISALIGNMENT.ENABLED:
            aug_list_after.append(MisAlignment(displacement=cfg.AUGMENTOR.MISALIGNMENT.DISPLACEMENT,
                            rotate_ratio=cfg.AUGMENTOR.MISALIGNMENT.ROTATE_RATIO,
                            p=cfg.AUGMENTOR.MISALIGNMENT.P,
                            additional_targets={'label': 'mask'}))
        if cfg.AUGMENTOR.MISSINGSECTION.ENABLED and not cfg.DATASET.DO_2D:
            aug_list_after.append(
                MissingSection(
                    num_sections=cfg.AUGMENTOR.MISSINGSECTION.NUM_SECTION,
                    p=cfg.AUGMENTOR.MISSINGSECTION.P,
                    additional_targets={'label': 'mask'}))
        augmentor_after = Compose(transforms=aug_list_after,
                            input_size=cfg.MODEL.INPUT_SIZE,
                            smooth=cfg.AUGMENTOR.SMOOTH,
                            keep_uncropped=keep_uncropped,
                            keep_non_smoothed=keep_non_smoothed,
                            additional_targets={'label': 'mask'})
        augmentor_before = Compose(transforms=aug_list,
                            input_size=augmentor_after.sample_size,
                            smooth=cfg.AUGMENTOR.SMOOTH,
                            keep_uncropped=keep_uncropped,
                            keep_non_smoothed=keep_non_smoothed,
                            additional_targets=additional_targets)
        return {'augmentor_before': augmentor_before, 'augmentor_after': augmentor_after}
    # else:
    #     # sample a larger patch without augmentation for augmentations in ssl
    #     sample_size = cfg.MODEL.INPUT_SIZE
    #     expand = [20, 100, 100]
    #     sample_size = [i + j for i, j in zip(sample_size, expand)]
    #     augmentor = Compose(transforms=[],
    #                         input_size=sample_size,
    #                         smooth=cfg.AUGMENTOR.SMOOTH,
    #                         keep_uncropped=keep_uncropped,
    #                         keep_non_smoothed=keep_non_smoothed,
    #                         additional_targets=additional_targets)

    return augmentor
    # if cfg.AUGMENTOR.USE_COPY_PASTE:
    #     return augmentor, CopyPasteAugmentor(cfg.AUGMENTOR.COPY_PASTE_THRES)
    # else:
    #     return augmentor

def build_ssl_augmentor(cfg):
    R"""Build the data augmentor for semi-supervised learning.
    """
    aug_list = []

    #1. rotate
    if cfg.AUGMENTOR.ROTATE.ENABLED:
        aug_list.append(
            Rotate(rot90=True,
                   p=cfg.AUGMENTOR.ROTATE.P))

    #2. flip
    if cfg.AUGMENTOR.FLIP.ENABLED:
        aug_list.append(
            Flip(do_ztrans=cfg.AUGMENTOR.FLIP.DO_ZTRANS,
                 p=cfg.AUGMENTOR.FLIP.P))

    #3. grayscale
    if cfg.AUGMENTOR.GRAYSCALE.ENABLED:
        aug_list.append(
            Grayscale(p=cfg.AUGMENTOR.GRAYSCALE.P))

    #4. missingparts
    if cfg.AUGMENTOR.MISSINGPARTS.ENABLED:
        aug_list.append(
            MissingParts(p=cfg.AUGMENTOR.MISSINGPARTS.P))

    #5. motion-blur
    if cfg.AUGMENTOR.MOTIONBLUR.ENABLED:
        aug_list.append(
            MotionBlur( 
                sections=cfg.AUGMENTOR.MOTIONBLUR.SECTIONS, 
                kernel_size=cfg.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE,
                p=cfg.AUGMENTOR.MOTIONBLUR.P))

    #6. cut-blur
    if cfg.AUGMENTOR.CUTBLUR.ENABLED:
        aug_list.append(
            CutBlur(length_ratio=cfg.AUGMENTOR.CUTBLUR.LENGTH_RATIO, 
                    down_ratio_min=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN,
                    down_ratio_max=cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX,
                    downsample_z=cfg.AUGMENTOR.CUTBLUR.DOWNSAMPLE_Z,
                    p=cfg.AUGMENTOR.CUTBLUR.P))

    #7. cut-noise
    if cfg.AUGMENTOR.CUTNOISE.ENABLED:
        aug_list.append(
            CutNoise(length_ratio=cfg.AUGMENTOR.CUTNOISE.LENGTH_RATIO, 
                     scale=cfg.AUGMENTOR.CUTNOISE.SCALE,
                     p=cfg.AUGMENTOR.CUTNOISE.P))

    return Compose(transforms=aug_list, 
                   input_size=cfg.MODEL.INPUT_SIZE, 
                   smooth=cfg.AUGMENTOR.SMOOTH,
                   additional_targets=None)
