from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


def commit_model(identifier):
    return ModelCommitment(identifier=identifier,
                           activations_model=get_model(identifier),
                           layers=get_layers(identifier))

model_registry['yudixie_resnet18_distance_reg_0_240719'] = lambda: commit_model('yudixie_resnet18_distance_reg_0_240719')
model_registry['yudixie_resnet18_translation_reg_0_240719'] = lambda: commit_model('yudixie_resnet18_translation_reg_0_240719')
model_registry['yudixie_resnet18_rotation_reg_0_240719'] = lambda: commit_model('yudixie_resnet18_rotation_reg_0_240719')
model_registry['yudixie_resnet18_distance_translation_0_240719'] = lambda: commit_model('yudixie_resnet18_distance_translation_0_240719')
model_registry['yudixie_resnet18_distance_rotation_0_240719'] = lambda: commit_model('yudixie_resnet18_distance_rotation_0_240719')
model_registry['yudixie_resnet18_translation_rotation_0_240719'] = lambda: commit_model('yudixie_resnet18_translation_rotation_0_240719')
model_registry['yudixie_resnet18_distance_translation_rotation_0_240719'] = lambda: commit_model('yudixie_resnet18_distance_translation_rotation_0_240719')
model_registry['yudixie_resnet18_category_class_0_240719'] = lambda: commit_model('yudixie_resnet18_category_class_0_240719')
model_registry['yudixie_resnet18_object_class_0_240719'] = lambda: commit_model('yudixie_resnet18_object_class_0_240719')
model_registry['yudixie_resnet18_cat_obj_class_all_latents_0_240719'] = lambda: commit_model('yudixie_resnet18_cat_obj_class_all_latents_0_240719')
model_registry['yudixie_resnet18_imagenet1kpret_0_240719'] = lambda: commit_model('yudixie_resnet18_imagenet1kpret_0_240719')
model_registry['yudixie_resnet18_random_0_240719'] = lambda: commit_model('yudixie_resnet18_random_0_240719')
