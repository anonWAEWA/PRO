"""Some prompts here are adapted from WISE-FT and LAION-AI Benchmark
https://github.com/mlfoundations/wise-ft/blob/58b7a4b343b09dc06606aa929c2ef51accced8d1/src/templates/utils.py#L25
https://github.com/LAION-AI/CLIP_benchmark

Other prompts are taken from openai's clip repository
https://github.com/openai/CLIP/blob/main/data/prompts.md
"""


def append_proper_article(name):
    name = name.replace("_", " ")
    if name[0] in "aeiou":
        return "an " + name
    return "a " + name


TEMPLATES = {
    "default": [
        lambda c: f"a bad photo of a {c}.",
        lambda c: f"a photo of many {c}.",
        lambda c: f"a sculpture of a {c}.",
        lambda c: f"a photo of the hard to see {c}.",
        lambda c: f"a low resolution photo of the {c}.",
        lambda c: f"a rendering of a {c}.",
        lambda c: f"graffiti of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a cropped photo of the {c}.",
        lambda c: f"a tattoo of a {c}.",
        lambda c: f"the embroidered {c}.",
        lambda c: f"a photo of a hard to see {c}.",
        lambda c: f"a bright photo of a {c}.",
        lambda c: f"a photo of a clean {c}.",
        lambda c: f"a photo of a dirty {c}.",
        lambda c: f"a dark photo of the {c}.",
        lambda c: f"a drawing of a {c}.",
        lambda c: f"a photo of my {c}.",
        lambda c: f"the plastic {c}.",
        lambda c: f"a photo of the cool {c}.",
        lambda c: f"a close-up photo of a {c}.",
        lambda c: f"a black and white photo of the {c}.",
        lambda c: f"a painting of the {c}.",
        lambda c: f"a painting of a {c}.",
        lambda c: f"a pixelated photo of the {c}.",
        lambda c: f"a sculpture of the {c}.",
        lambda c: f"a bright photo of the {c}.",
        lambda c: f"a cropped photo of a {c}.",
        lambda c: f"a plastic {c}.",
        lambda c: f"a photo of the dirty {c}.",
        lambda c: f"a jpeg corrupted photo of a {c}.",
        lambda c: f"a blurry photo of the {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a good photo of the {c}.",
        lambda c: f"a rendering of the {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"a photo of one {c}.",
        lambda c: f"a doodle of a {c}.",
        lambda c: f"a close-up photo of the {c}.",
        lambda c: f"a photo of a {c}.",
        lambda c: f"the origami {c}.",
        lambda c: f"the {c} in a video game.",
        lambda c: f"a sketch of a {c}.",
        lambda c: f"a doodle of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a low resolution photo of a {c}.",
        lambda c: f"the toy {c}.",
        lambda c: f"a rendition of the {c}.",
        lambda c: f"a photo of the clean {c}.",
        lambda c: f"a photo of a large {c}.",
        lambda c: f"a rendition of a {c}.",
        lambda c: f"a photo of a nice {c}.",
        lambda c: f"a photo of a weird {c}.",
        lambda c: f"a blurry photo of a {c}.",
        lambda c: f"a cartoon {c}.",
        lambda c: f"art of a {c}.",
        lambda c: f"a sketch of the {c}.",
        lambda c: f"a embroidered {c}.",
        lambda c: f"a pixelated photo of a {c}.",
        lambda c: f"itap of the {c}.",
        lambda c: f"a jpeg corrupted photo of the {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a plushie {c}.",
        lambda c: f"a photo of the nice {c}.",
        lambda c: f"a photo of the small {c}.",
        lambda c: f"a photo of the weird {c}.",
        lambda c: f"the cartoon {c}.",
        lambda c: f"art of the {c}.",
        lambda c: f"a drawing of the {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a black and white photo of a {c}.",
        lambda c: f"the plushie {c}.",
        lambda c: f"a dark photo of a {c}.",
        lambda c: f"itap of a {c}.",
        lambda c: f"graffiti of the {c}.",
        lambda c: f"a toy {c}.",
        lambda c: f"itap of my {c}.",
        lambda c: f"a photo of a cool {c}.",
        lambda c: f"a photo of a small {c}.",
        lambda c: f"a tattoo of the {c}.",
    ],
    "cifar10": [
        lambda c: f"a photo of {c}",
        lambda c: f"a blurry photo of a {c}",
        lambda c: f"a black and white photo of a {c}",
        lambda c: f"a low contrast photo of a {c}",
        lambda c: f"a high contrast photo of a {c}",
        lambda c: f"a bad photo of a {c}",
        lambda c: f"a good photo of a {c}",
        lambda c: f"a photo of a small {c}",
        lambda c: f"a photo of a big {c}",
        lambda c: f"a photo of the {c}",
        lambda c: f"a blurry photo of the {c}",
        lambda c: f"a black and white photo of the {c}",
        lambda c: f"a low contrast photo of the {c}",
        lambda c: f"a high contrast photo of the {c}",
        lambda c: f"a bad photo of the {c}",
        lambda c: f"a good photo of the {c}",
        lambda c: f"a photo of the small {c}",
        lambda c: f"a photo of the big {c}",
    ],
    "cifar100": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a blurry photo of a {c}.",
        lambda c: f"a black and white photo of a {c}.",
        lambda c: f"a low contrast photo of a {c}.",
        lambda c: f"a high contrast photo of a {c}.",
        lambda c: f"a bad photo of a {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a photo of a small {c}.",
        lambda c: f"a photo of a big {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a blurry photo of the {c}.",
        lambda c: f"a black and white photo of the {c}.",
        lambda c: f"a low contrast photo of the {c}.",
        lambda c: f"a high contrast photo of the {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a good photo of the {c}.",
        lambda c: f"a photo of the small {c}.",
        lambda c: f"a photo of the big {c}.",
    ],
    "svhn": [
        lambda c: f"a photo of the number {c} written on a sign",
        lambda c: f"an outdoor house number {c}",
        lambda c: f"the number {c} in the center of the image",
        lambda c: f"an outdoor number {c} writte on a sign",
        lambda c: f"an outdoor number {c}",
        lambda c: f"a centered image of the number {c}",
    ],
    "eurosat": [
        lambda c: f"a centered satellite photo of {c}.",
        lambda c: f"a centered satellite photo of a {c}.",
        lambda c: f"a centered satellite photo of the {c}.",
    ],
    "mnist": [lambda c: f'a photo of the number: "{c}".',],
    "camelyon": [lambda c: f'a microscopic image of "{c}".',],
    "fmow": [
        lambda c: f"satellite photo of a {c}.",
        lambda c: f"aerial photo of a {c}.",
        lambda c: f"satellite photo of {append_proper_article(c)}.",
        lambda c: f"aerial photo of {append_proper_article(c)}.",
        lambda c: f"satellite photo of a {c} in asia.",
        lambda c: f"aerial photo of a {c} in asia.",
        lambda c: f"satellite photo of a {c} in africa.",
        lambda c: f"aerial photo of a {c} in africa.",
        lambda c: f"satellite photo of a {c} in the americas.",
        lambda c: f"aerial photo of a {c} in the americas.",
        lambda c: f"satellite photo of a {c} in europe.",
        lambda c: f"aerial photo of a {c} in europe.",
        lambda c: f"satellite photo of a {c} in oceania.",
        lambda c: f"aerial photo of a {c} in oceania.",
        lambda c: f"a photo of a {c}.",
        lambda c: f"{c}.",
    ],
    "food101": [lambda c: f"a photo of {c}, a type of food."],
    "caltech101": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a painting of a {c}.",
        lambda c: f"a plastic {c}.",
        lambda c: f"a sculpture of a {c}.",
        lambda c: f"a sketch of a {c}.",
        lambda c: f"a tattoo of a {c}.",
        lambda c: f"a toy {c}.",
        lambda c: f"a rendition of a {c}.",
        lambda c: f"a embroidered {c}.",
        lambda c: f"a cartoon {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"a plushie {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"art of a {c}.",
        lambda c: f"graffiti of a {c}.",
        lambda c: f"a drawing of a {c}.",
        lambda c: f"a doodle of a {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a painting of the {c}.",
        lambda c: f"the plastic {c}.",
        lambda c: f"a sculpture of the {c}.",
        lambda c: f"a sketch of the {c}.",
        lambda c: f"a tattoo of the {c}.",
        lambda c: f"the toy {c}.",
        lambda c: f"a rendition of the {c}.",
        lambda c: f"the embroidered {c}.",
        lambda c: f"the cartoon {c}.",
        lambda c: f"the {c} in a video game.",
        lambda c: f"the plushie {c}.",
        lambda c: f"the origami {c}.",
        lambda c: f"art of the {c}.",
        lambda c: f"graffiti of the {c}.",
        lambda c: f"a drawing of the {c}.",
        lambda c: f"a doodle of the {c}.",
    ],
    "oxfordpet": [lambda c: f"a photo of a {c}, a type of pet.",],
    "flowers102": [lambda c: f"a photo of a {c}, a type of flower.",],
    "stanfordcars": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a photo of my {c}.",
        lambda c: f"i love my {c}!",
        lambda c: f"a photo of my dirty {c}.",
        lambda c: f"a photo of my clean {c}.",
        lambda c: f"a photo of my new {c}.",
        lambda c: f"a photo of my old {c}.",
    ],
    "country211": [
        lambda c: f"a photo i took in {c}.",
        lambda c: f"a photo i took while visiting {c}.",
        lambda c: f"a photo from my home country of {c}.",
        lambda c: f"a photo from my visit to {c}.",
        lambda c: f"a photo showing the country of {c}.",
    ],
    "sst2": [lambda c: f"a {c} review of a movie."],
    "sun397": [lambda c: f"a photo of a {c}.", lambda c: f"a photo of the {c}.",],
    "stl10": [lambda c: f"a photo of a {c}.", lambda c: f"a photo of the {c}.",],
    "fer2013": [
        lambda c: f"a photo of a {c} looking face.",
        lambda c: f"a photo of a face showing the emotion: {c}.",
        lambda c: f"a photo of a face looking {c}.",
        lambda c: f"a face that looks {c}.",
        lambda c: f"they look {c}.",
        lambda c: f"look at how {c} they are.",
    ],
    "imagenet": [
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"art of the {c}.",
        lambda c: f"a photo of the small {c}.",
    ],
    "voc2007": [lambda c: f"a photo of a {c}.",],
    "fgvcaircraft": [
        lambda c: f"a photo of a {c}, a type of aircraft.",
        lambda c: f"a photo of the {c}, a type of aircraft.",
    ],
    "resisc45": [
        lambda c: f"satellite imagery of {c}.",
        lambda c: f"aerial imagery of {c}.",
        lambda c: f"satellite photo of {c}.",
        lambda c: f"aerial photo of {c}.",
        lambda c: f"satellite view of {c}.",
        lambda c: f"aerial view of {c}.",
        lambda c: f"satellite imagery of a {c}.",
        lambda c: f"aerial imagery of a {c}.",
        lambda c: f"satellite photo of a {c}.",
        lambda c: f"aerial photo of a {c}.",
        lambda c: f"satellite view of a {c}.",
        lambda c: f"aerial view of a {c}.",
        lambda c: f"satellite imagery of the {c}.",
        lambda c: f"aerial imagery of the {c}.",
        lambda c: f"satellite photo of the {c}.",
        lambda c: f"aerial photo of the {c}.",
        lambda c: f"satellite view of the {c}.",
        lambda c: f"aerial view of the {c}.",
    ],
    "dtd": [
        lambda c: f"a photo of a {c} texture.",
        lambda c: f"a photo of a {c} pattern.",
        lambda c: f"a photo of a {c} thing.",
        lambda c: f"a photo of a {c} object.",
        lambda c: f"a photo of the {c} texture.",
        lambda c: f"a photo of the {c} pattern.",
        lambda c: f"a photo of the {c} thing.",
        lambda c: f"a photo of the {c} object.",
    ],
    "gtsrb": [
        lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
        lambda c: f'a centered photo of a "{c}" traffic sign.',
        lambda c: f'a close up photo of a "{c}" traffic sign.',
    ],
    "pcam": [lambda c: f"this is a photo of {c}",],
    "kitti": [lambda c: f"{c}",],
    "birdsnap": [lambda c: f"a photo of a {c}, a type of bird.",],
    "ucf101": [
        lambda c: f"a photo of a person {c}.",
        lambda c: f"a video of a person {c}.",
        lambda c: f"a example of a person {c}.",
        lambda c: f"a demonstration of a person {c}.",
        lambda c: f"a photo of the person {c}.",
        lambda c: f"a video of the person {c}.",
        lambda c: f"a example of the person {c}.",
        lambda c: f"a demonstration of the person {c}.",
        lambda c: f"a photo of a person using {c}.",
        lambda c: f"a video of a person using {c}.",
        lambda c: f"a example of a person using {c}.",
        lambda c: f"a demonstration of a person using {c}.",
        lambda c: f"a photo of the person using {c}.",
        lambda c: f"a video of the person using {c}.",
        lambda c: f"a example of the person using {c}.",
        lambda c: f"a demonstration of the person using {c}.",
        lambda c: f"a photo of a person doing {c}.",
        lambda c: f"a video of a person doing {c}.",
        lambda c: f"a example of a person doing {c}.",
        lambda c: f"a demonstration of a person doing {c}.",
        lambda c: f"a photo of the person doing {c}.",
        lambda c: f"a video of the person doing {c}.",
        lambda c: f"a example of the person doing {c}.",
        lambda c: f"a demonstration of the person doing {c}.",
        lambda c: f"a photo of a person during {c}.",
        lambda c: f"a video of a person during {c}.",
        lambda c: f"a example of a person during {c}.",
        lambda c: f"a demonstration of a person during {c}.",
        lambda c: f"a photo of the person during {c}.",
        lambda c: f"a video of the person during {c}.",
        lambda c: f"a example of the person during {c}.",
        lambda c: f"a demonstration of the person during {c}.",
        lambda c: f"a photo of a person performing {c}.",
        lambda c: f"a video of a person performing {c}.",
        lambda c: f"a example of a person performing {c}.",
        lambda c: f"a demonstration of a person performing {c}.",
        lambda c: f"a photo of the person performing {c}.",
        lambda c: f"a video of the person performing {c}.",
        lambda c: f"a example of the person performing {c}.",
        lambda c: f"a demonstration of the person performing {c}.",
        lambda c: f"a photo of a person practicing {c}.",
        lambda c: f"a video of a person practicing {c}.",
        lambda c: f"a example of a person practicing {c}.",
        lambda c: f"a demonstration of a person practicing {c}.",
        lambda c: f"a photo of the person practicing {c}.",
        lambda c: f"a video of the person practicing {c}.",
        lambda c: f"a example of the person practicing {c}.",
        lambda c: f"a demonstration of the person practicing {c}.",
    ],
}
