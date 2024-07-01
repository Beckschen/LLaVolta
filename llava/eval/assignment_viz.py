from PIL import Image
import torch
def assignment_viz(image,assignment):
    '''
    image: 1 x 3 x 336 x 336
    assignment 1 x 72 x 576
    '''
    image = image.resize((336,336))
    imglist = []
    if assignment is None:
        return None
    assignment = assignment.squeeze(0).transpose(0,1)
    for i in range(assignment.shape[0]):
        #blend the image with the assignment
        cur_assignment = assignment[i] # 576
        cur_assignment = cur_assignment.view(24,24).unsqueeze(-1).repeat(1,1,3)
        cur_assignment = cur_assignment.cpu().detach().numpy()
        cur_assignment = cur_assignment - cur_assignment.min()
        cur_assignment = cur_assignment / cur_assignment.max()
        cur_assignment = cur_assignment * 255
        cur_assignment = cur_assignment.astype('uint8')
        cur_assignment = Image.fromarray(cur_assignment,'RGB')
        cur_assignment = cur_assignment.resize((336,336))
        cur_image = Image.blend(image,cur_assignment,0.5)
        imglist.append(cur_image)
    return imglist