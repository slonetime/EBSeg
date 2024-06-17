from detectron2.config import CfgNode as CN


def add_ebseg_config(cfg):
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP = [
        "absolute_pos_embed",
        "positional_embedding",
        "pos_embed",
        "query_embed",
        "relative_position_bias_table",
    ]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CLIP_MULTIPLIER = 1.0
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1

    cfg.SOLVER.TEST_TIME_TRAINER = False

    cfg.MODEL.EBSEG = CN()
    cfg.MODEL.EBSEG.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.EBSEG.CLASS_WEIGHT = 2.0
    cfg.MODEL.EBSEG.DICE_WEIGHT = 5.0
    cfg.MODEL.EBSEG.MASK_WEIGHT = 5.0
    cfg.MODEL.EBSEG.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.EBSEG.NUM_CLASSES = 171
    cfg.MODEL.EBSEG.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.EBSEG.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.EBSEG.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.EBSEG.CLIP_PRETRAINED_NAME = "openai"
    cfg.MODEL.EBSEG.CLIP_TEMPLATE_SET = "vild"
    cfg.MODEL.EBSEG.FEATURE_LAST_LAYER_IDX = 9
    cfg.MODEL.EBSEG.HEAD_FIRST_LAYER_IDX = 9
    cfg.MODEL.EBSEG.CLIP_FROZEN_EXCLUDE = ["positional_embedding"]
    cfg.MODEL.EBSEG.SIZE_DIVISIBILITY = 32
    cfg.MODEL.EBSEG.ASYMETRIC_INPUT = True
    cfg.MODEL.EBSEG.CLIP_RESOLUTION = 0.5
    cfg.MODEL.EBSEG.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True
    
    cfg.MODEL.SAM = CN()
    cfg.MODEL.SAM.MODEL_NAME = 'vit_b'
    
    cfg.MODEL.SAN_HEAD = CN()
    cfg.MODEL.SAN_HEAD.NUM_QUERIES = 100
    cfg.MODEL.SAN_HEAD.CLIP_DEEPER_FROZEN_EXCLUDE = []
    cfg.MODEL.SAN_HEAD.SOS_TOKEN_FORMAT = "cls_token"
    cfg.MODEL.SAN_HEAD.REC_DOWNSAMPLE_METHOD = "max"

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "EBSeg"
    cfg.WANDB.NAME = None
    # use flash attention
    cfg.MODEL.FLASH = False

    cfg.MODEL.MODIFIED = CN()
    cfg.MODEL.MODIFIED.SCOREMAP_LOSS_WEIGHT = 1.0
    cfg.MODEL.MODIFIED.FUSION_ENCODER_LAYERS = 1
    cfg.MODEL.MODIFIED.OVCLASSFIER_TYPE = 0
    cfg.MODEL.MODIFIED.PROMPT_TRAINING = False
    cfg.MODEL.MODIFIED.LOAD_PROMPT_PATH = ''
    cfg.MODEL.MODIFIED.WITH_SCOREMAP_BRANCH = True
    cfg.MODEL.MODIFIED.WITH_AGG_LAYER = False
    cfg.MODEL.MODIFIED.WITH_TEXT_FUSION = False
    cfg.MODEL.MODIFIED.ONLY_SCOREMAP_BRANCH_TRAINING = True
    cfg.MODEL.MODIFIED.PROMPT_TRAINING = False
    cfg.MODEL.MODIFIED.TEXT_DIVERSIFICATION = False
    cfg.MODEL.MODIFIED.SSC_LOSS = None


    cfg.MODEL.MASK_FORMER = CN()
    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75