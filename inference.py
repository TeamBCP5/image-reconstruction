from tqdm import tqdm
import cv2
import numpy as np
import torch

def inference(model, img_paths, img_size=512, stride=512, transforms=None, device=None):
    results = []
    for img_path, label_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms(image=img)['image']

        crop = []
        position = []

        result_img = np.zeros_like(img.numpy().transpose(1,2,0))
        voting_mask = np.zeros_like(img.numpy().transpose(1,2,0))
        
        img = img.unsqueeze(0)
        for top in range(0, img.shape[2], stride):
            for left in range(0, img.shape[3], stride):
                piece = torch.zeros([1, 3, img_size, img_size])
                temp = img[:, :, top:top+img_size, left:left+img_size]
                piece[:, :, :temp.shape[2], :temp.shape[3]] = temp
                with torch.no_grad():
                    pred = model(piece.to(device))
                pred = pred[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = (pred*127.5)+127.5
                crop.append(pred)
                position.append([top, left])

        crop = np.array(crop).astype(np.uint16)
        for num, (t, l) in enumerate(position):
            piece = crop[num]
            h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
            result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w, :]
            voting_mask[t:t+img_size, l:l+img_size, :] += 1
        
        result_img = result_img / voting_mask
        result_img = result_img.astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        results.append(result_img)

    return results