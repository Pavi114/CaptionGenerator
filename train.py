from constants import EPOCHS
from torchvision import transforms
from main_model import CaptionGenerator

transform = transforms.Compose([ 
    transforms.Resize((256, 256)),                          
    transforms.RandomCrop(224),                     
    transforms.RandomHorizontalFlip(),              
    transforms.ToTensor(),                          
    transforms.Normalize((0.485, 0.456, 0.406),  
                         (0.229, 0.224, 0.225))]
)

caption_generator = CaptionGenerator(
    '../COCO/', 
    './captions.csv', 
    'annotations_trainval2014/annotations/instances_train2014.json',
    'train/images/',
    transform=transform
)

## comment while predicting
caption_generator.train(EPOCHS)

### to fill
image_path = ''
model_path = ''
caption_generator.load_model(model_path)
caption_generator.predict_using_sampling(image_path)
