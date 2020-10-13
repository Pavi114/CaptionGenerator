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
    '/media/aananth/OS/Users/aanan/Documents/COCO/', 
    '/media/aananth/OS/Users/aanan/Documents/COCO/captions.csv', 
    'instances_train2014.json',
    'train/images/',
    transform=transform
)

# Load model
model_path = '/home/aananth/dev/CaptionGenerator/models/epoch-2.pkl'
caption_generator.load_model(model_path)

## comment while predicting
# caption_generator.train(EPOCHS)

### to fill
image_path = '/home/aananth/dev/CaptionGenerator/61-l8S81xVL._SL1500_.jpg'
caption_generator.predict_using_sampling(image_path)

image_path = '/home/aananth/dev/CaptionGenerator/1.jpg'
caption_generator.predict_using_sampling(image_path)
