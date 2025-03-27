# Glct: A Novel Global-Local Constraint  for Unpaired Image-to-Image Translation

### Basic Usage

- Training:
```bash
python train.py --dataroot=datasets/cityscapes --direction=AtoB
```
- Test:
put the trained checkpoints to the folder ```checkpoints/cityscapes```
```bash
python test.py --dataroot=datasets/cityscapes --name=cityscapes --direction=AtoB
```




