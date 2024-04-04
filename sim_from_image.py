from PIL import Image

image_path = 'thyme_img.png'
out_dir = 'thyme_xy'

img = Image.open(image_path).convert('LA') 
 

for i in range(250):
    left = i
    top = i
    img_res = img.crop((left, top, left+512, top+512))
    leading_zeros = ''.join([str(0) for _ in range(4-len(str(i+1)))])
    img_res.save(f'{out_dir}\\{leading_zeros}{i+1}.png', 'PNG')