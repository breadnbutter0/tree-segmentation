import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image
from rs3 import LightImage, save_map
from normalization import norm_min_max
from tqdm import tqdm
import pickle
from osgeo import gdal
import pyproj
import pandas as pd
import os


class Segmentation():
    
    def __init__(self, img_fn):
        
        # sam load data
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        # sam automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        # sam predictor
        self.predictor = SamPredictor(sam)
        
        # image metadata
        self.img = LightImage(img_fn)

    def point_prompt_segmentation(self, point_prompt_fn, mask_point_fn, bbox_point_fn, max_tree_size=2000, buffer=5):

        # load point prompts
        with open(point_prompt_fn,'rb') as f:
            point_prompts = pickle.load(f)

        # segmentation
        input_label = np.array([1])

        i=1
        bbox = {'id':[], 'bbox':[]}
        mask_from_point = np.zeros((self.img.nrow,self.img.ncol))
        for x,y in tqdm(point_prompts):
            input_point = np.array([[max_tree_size/2, max_tree_size/2]])
            xmin,xmax = int(x-max_tree_size/2),int(x+max_tree_size/2)
            ymin,ymax = int(y-max_tree_size/2),int(y+max_tree_size/2)
            
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > self.img.ncol:
                xmax = self.img.ncol
            if ymax > self.img.nrow:
                ymax = self.img.nrow
            
            r = self.img.get_box(xmin,xmax,ymin,ymax, band=0)
            g = self.img.get_box(xmin,xmax,ymin,ymax, band=1)
            b = self.img.get_box(xmin,xmax,ymin,ymax, band=2)
            img_rgb = np.array([r,g,b]).transpose([1,2,0])
            img_uint8 = np.array(norm_min_max(img_rgb, new_min=0, new_max=255), dtype=np.uint8)
            self.predictor.set_image(img_uint8)
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True)
            best_mask = masks[np.argmax(scores),:,:]
            rows = np.any(best_mask, axis=1)
            cols = np.any(best_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            if x_min > buffer and x_max < max_tree_size-buffer and y_min > buffer and y_min < max_tree_size - buffer:
                mask_subset = mask_from_point[ymin:ymax,xmin:xmax]
                mask_nonoverlay = ~(mask_subset>0) * best_mask
                if mask_nonoverlay.any():
                    mask_from_point[ymin:ymax,xmin:xmax] += mask_nonoverlay*i
                    rows = np.any(mask_nonoverlay, axis=1)
                    cols = np.any(mask_nonoverlay, axis=0)
                    y, y_max = np.where(rows)[0][[0, -1]]
                    x, x_max = np.where(cols)[0][[0, -1]]
                    w = x_max - x
                    h = y_max - y
                    bbox['id'].append(i)
                    bbox['bbox'].append([x+xmin,x+xmin+w,y+ymin,y+ymin+h])            
                    i+=1
                   
        with open(mask_point_fn,'wb') as f:
            pickle.dump(mask_from_point, f)
            
        with open(bbox_point_fn,'wb') as f:
            pickle.dump(bbox, f)

    def automatic_segmentation(self, mask_automatic_fn, point_prompt_fn, bbox_automatic_fn, crop_size=5000, buffer=100):
        
        resize = 1024
        scale = crop_size/resize
        nrow = int(self.img.nrow // crop_size)
        ncol = int(self.img.ncol // crop_size)

        bbox = {'id':[], 'bbox':[]}
        point_prompts=[]
        mask_all = np.zeros((self.img.nrow,self.img.ncol), dtype=np.int32)
        k=1
        for n in tqdm(range(nrow*ncol)):
            i = n // ncol
            j = n % ncol
            minx, maxx = int(j*crop_size), int(j*crop_size + crop_size)
            miny, maxy = int(i*crop_size), int(i*crop_size + crop_size)
            
            if maxx > self.img.ncol:
                maxx = self.img.ncol
            
            if maxy > self.img.nrow:
                maxy = self.img.nrow      
            
            r = self.img.get_box(minx,maxx,miny,maxy, band=0)
            g = self.img.get_box(minx,maxx,miny,maxy, band=1)
            b = self.img.get_box(minx,maxx,miny,maxy, band=2)
            # nir = img.get_box(minx,maxx,miny,maxy, band=6)
            img_rgb = np.array([r,g,b]).transpose([1,2,0])
            img_uint8 = np.array(norm_min_max(img_rgb, new_min=0, new_max=255), dtype=np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_resize = np.array(img_pil.resize((resize,resize)))
            masks = self.mask_generator.generate(img_resize)
            mask_arr = np.zeros((maxy-miny,maxx-minx), dtype=np.int32)
            for mask in masks:
                x,y,w,h = mask['bbox']
                if x < buffer or x+w > resize-buffer or y < buffer or y+h > resize-buffer:
                    point_x, point_y = mask['point_coords'][0]
                    point_prompts.append([scale*point_x+minx, scale*point_y+miny])
                else:
                    mask_resize = np.array(Image.fromarray(mask['segmentation']).resize((maxx-minx,maxy-miny)), dtype=np.int32)
                    mask_nonoverlay = ~(mask_arr>0) * mask_resize
                    if mask_nonoverlay.any():
                        mask_arr +=  mask_nonoverlay * k
                        rows = np.any(mask_nonoverlay, axis=1)
                        cols = np.any(mask_nonoverlay, axis=0)
                        y, y_max = np.where(rows)[0][[0, -1]]
                        x, x_max = np.where(cols)[0][[0, -1]]
                        w = x_max - x
                        h = y_max - y
                        bbox['id'].append(k)
                        bbox['bbox'].append([x+minx,x+minx+w,y+miny,y+miny+h])            
                        k+=1
            mask_all[miny:maxy,minx:maxx]=mask_arr

            
        with open(mask_automatic_fn,'wb') as f:
            pickle.dump(mask_all, f)
            
        with open(point_prompt_fn,'wb') as f:
            pickle.dump(point_prompts, f)
            
        with open(bbox_automatic_fn,'wb') as f:
            pickle.dump(bbox, f)
            
    
    def do_segmentation(self, mask_automatic_fn, point_prompt_fn, bbox_automatic_fn,
                        mask_point_prompt_fn, bbox_point_prompt_fn, mask_all_fn, bbox_all_fn, eType=gdal.GDT_Int32):
        
        print('automatic segmentation..')
        self.automatic_segmentation(
                                mask_automatic_fn=mask_automatic_fn,
                                point_prompt_fn=point_prompt_fn,
                                bbox_automatic_fn=bbox_automatic_fn
                                )
        
        print('point prompt segmentation..')
        self.point_prompt_segmentation(
                                   point_prompt_fn=point_prompt_fn,
                                   mask_point_fn=mask_point_prompt_fn,
                                   bbox_point_fn=bbox_point_prompt_fn
                                   )
        
        print('merging two mask files..')
        self.merge_automatic_point(self, mask_automatic_fn,  bbox_automatic_fn,
                        mask_point_prompt_fn, bbox_point_prompt_fn, )
        
        print('saving files..')
        self.save_output(self, mask_all_fn, bbox_all_fn, eType=gdal.GDT_Int32)


    def merge_automatic_point(self, mask_automatic_fn,  bbox_automatic_fn,
                        mask_point_prompt_fn, bbox_point_prompt_fn):
      
        # merge two mask files
        with open(mask_automatic_fn,'rb') as f:
            mask_automatic = pickle.load(f)
            
        with open(mask_point_prompt_fn,'rb') as f:
            mask_from_point = pickle.load(f)
        
        mask_from_point_nonoverlay = np.array((~(mask_automatic>0)) * mask_from_point, dtype = np.int32)
        max_id = np.max(mask_automatic)
        mask_from_point_nonoverlay[mask_from_point_nonoverlay>0] += max_id
        mask_tree_all = mask_automatic + mask_from_point_nonoverlay


        print('creating bounding boxes..')
        # merge two bbox files
        with open(bbox_automatic_fn, 'rb') as f:
            bbox_automatic = pickle.load(f)

        with open(bbox_point_prompt_fn, 'rb') as f:
            bbox_point_prompt = pickle.load(f)

        bbox_all = {'id':[], 'xmin':[], 'xmax':[], 'ymin':[], 'ymax':[]}
        bbox_all['id'] += (list(bbox_automatic['id']) + list(np.array(bbox_point_prompt['id']) + bbox_automatic['id'][-1]))
        bbox_all['xmin'] += (list(np.array(bbox_automatic['bbox'])[:,0]) + list(np.array(bbox_point_prompt['bbox'])[:,0]))
        bbox_all['xmax'] += (list(np.array(bbox_automatic['bbox'])[:,1]) + list(np.array(bbox_point_prompt['bbox'])[:,1]))
        bbox_all['ymin'] += (list(np.array(bbox_automatic['bbox'])[:,2]) + list(np.array(bbox_point_prompt['bbox'])[:,2]))
        bbox_all['ymax'] += (list(np.array(bbox_automatic['bbox'])[:,3]) + list(np.array(bbox_point_prompt['bbox'])[:,3]))

        df = pd.DataFrame(bbox_all)
        df_ordered = df.sort_values(by=['ymin','xmin'])

        print('Reordering label ids..')
        # Create an ordered mask based on the mapping provided in df_ordered
        id_to_index = {id: idx+1 for idx, id in enumerate(df_ordered['id'])}
        id_to_index[0] = 0

        # Create a lookup array where each value corresponds to its new position
        self.mask_all_ordered = np.zeros_like(mask_tree_all, dtype=int)

        for i in tqdm(range(self.img.nrow)):
            for j in range(self.img.ncol):
                self.mask_all_ordered[i,j] = id_to_index[mask_tree_all[i,j]]

        self.bbox_all_ordered = {
                    'id':bbox_all['id'], 
                    'xmin':df_ordered['xmin'],
                    'xmax':df_ordered['xmax'],
                    'ymin':df_ordered['ymin'],
                    'ymax':df_ordered['ymax'],
                    }
        

    def save_output(self, mask_all_fn, bbox_all_fn, eType=gdal.GDT_Int32):

        target_epsg = pyproj.CRS.from_wkt(self.img.projection.ExportToWkt()).to_epsg()
        
        save_map(map_fn = mask_all_fn, 
            src = self.mask_all_ordered, 
            ncol = self.img.ncol, 
            nrow = self.img.nrow, 
            target_epsg = target_epsg, 
            geotransform = self.img.geotransform, 
            eType = eType, 
            format = 'GTiff')

        with open(bbox_all_fn, 'wb') as f:
            pickle.dump(self.bbox_all_ordered,f)


def segmentation_bigtiff(args, automatic=False, point=False, all=False, merge=False, save=False):
    sam_pred = Segmentation(args[0])
    if automatic:
        sam_pred.automatic_segmentation(args[1], args[3], args[2])
    if point:
        sam_pred.point_prompt_segmentation(args[3], args[4], args[5])
    if all:
        sam_pred.do_segmentation(args[1], args[3], args[2], args[4], args[5], args[6], args[7])
    if merge:
        sam_pred.merge_automatic_point(args[1], args[2], args[4], args[5])
    if save:
        sam_pred.save_output(args[6], args[7])


if __name__=='__main__':

    img_fn = sys.argv[1]

    path = os.path.dirname(img_fn)
    fid = os.path.splitext(os.path.basename(img_fn))[0]

    mask_automatic_fn = f'{path}/{fid}_automatic_masks.pkl'
    bbox_automatic_fn = f'{path}/{fid}_bbox_automatic.pkl'

    point_prompt_fn = f'{path}/{fid}_point_prompts.pkl'

    mask_point_prompt_fn = f'{path}/{fid}_point_masks.pkl'
    bbox_point_prompt_fn = f'{path}/{fid}_bbox_point_prompt.pkl'

    mask_all_fn = f'{path}/{fid}_mask_all.tif'
    bbox_all_fn = f'{path}/{fid}_bbox_all.pkl'

    args=[img_fn, 
          mask_automatic_fn, bbox_automatic_fn, point_prompt_fn, 
          mask_point_prompt_fn, bbox_point_prompt_fn,
          mask_all_fn, bbox_all_fn]

    segmentation_bigtiff(args, all=True)