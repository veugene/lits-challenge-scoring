"""
Extracts volume and segmentations from MITK files.
Compares volumes to reference volumes. Checks for segmentation correctness.

RUN in directory containing the unzipped "from_google_drive" folder with 
manual segmentations and corrections (variability study).
"""


import os
import shutil
import numpy as np
import zipfile
import SimpleITK as sitk


dir_reference = "reference"
dir_volumes = "volumes"
dir_target = "processed_segmentations"
dir_source_manual = \
    ["from_google_drive/assia/correction_de_segmentation_automatique_1",
     "from_google_drive/assia/correction_de_segmentation_automatique_2",
     "from_google_drive/assia/segmentation_manuelle_1",
     "from_google_drive/assia/segmentation_manuelle_2",
     "from_google_drive/walid/correction_de_segmentation_automatique_1",
     "from_google_drive/walid/correction_de_segmentation_automatique_2",
     "from_google_drive/walid/segmentation_manuelle_1",
     "from_google_drive/walid/segmentation_manuelle_2"]
dir_source_auto = "automatic segmentations"
dir_temp = "/tmp/lits"


def extract(d_source, case):
    # Find all MITK files for this case.
    mitk_files = []
    for fn in sorted(os.listdir(os.path.join(d_source, case))):
        if fn.endswith(".mitk"):
            mitk_files.append(fn)
    if len(mitk_files)==0:
        print("ERROR: no MITK files found for case {} in {} -- SKIPPING!"
              "".format(case, d_source))
    if len(mitk_files)>1:
        print("WARNING: found more than one MITK file for case {} in {} " \
              "-- selecting first of :".format(case, d_source)
              + "\n".join(mitk_files))          
    print("PROCESSING {}".format(os.path.join(d_source, case, mitk_files[0])))
    
    # Extract MITK file.
    if os.path.exists(dir_temp):
        shutil.rmtree(dir_temp)
    os.makedirs(dir_temp)
    zip_ref = zipfile.ZipFile(os.path.join(d_source, case, mitk_files[0]), 'r')
    zip_ref.extractall(dir_temp)
    zip_ref.close()
    
    # Load reference volume.
    volume_ref_f = sitk.ReadImage(os.path.join(dir_volumes, case+"_CT.nii.gz"))
    volume_ref = sitk.GetArrayFromImage(volume_ref_f)
    
    # All masks are assumed to have only values of 0 and 1.
    def is_mask(arr):
        if np.any(arr>1) or np.any(arr<0):
            return False
        for i in np.unique(arr):
            if i not in [0, 1]:
                return False
        return True
        
    # Load volume.
    volume = None
    volume_f = None
    for fn in sorted(os.listdir(dir_temp)):
        if fn.endswith(".nrrd"):
            im_f = sitk.ReadImage(os.path.join(dir_temp, fn))
            im = sitk.GetArrayFromImage(im_f)
            if not is_mask(im):
                print("Found volume in {}".format(os.path.join(dir_temp, fn)))
                if volume is None:
                    volume = im
                    volume_f = im_f
                else:
                    print("ERROR: more than one volume found -- SKIPPING.")
                    return
                    
    # Hash volumes.
    volume_hash = hash(volume.tostring())
    volume_ref_hash = hash(volume_ref.tostring())
    if volume_hash!=volume_ref_hash:
        print("WARNING: reference and manual volume hashes do not match "
              "for {}/{}".format(d_source, case))
              
    # Crop or pad masks as needed.
    def process_mask(mask, name="a mask"):
        if np.count_nonzero(mask)==0:
            print("WARNING: {} is all zeros.".format(name))
        if mask.shape[1:]!=volume_ref.shape[1:]:
            print("ERROR: axial slice shape in {} differs from that in the "
                  "reference volume -- SKIPPING.".format(name))
            return
        if len(mask)<len(volume_ref):
            print("WARNING: {} has fewer slices than reference volume -- "
                  "zero-filling end section.".format(name))
            mask_fill = np.zeros(volume_ref.shape, dtype=volume_ref.dtype)
            mask_fill[:len(mask)] = mask[...]
            mask = mask_fill
        if len(mask)>len(volume_ref):
            print("WARNING: {} has more slices than reference volume -- "
                  "removing extra slices at the end.".format(name))
            mask = mask[:len(volume_ref)]
        return mask

    # Load masks.
    mask = None
    for fn in sorted(os.listdir(dir_temp)):
        if fn.endswith(".nrrd"):
            im_f = sitk.ReadImage(os.path.join(dir_temp, fn))
            im = sitk.GetArrayFromImage(im_f)
            if is_mask(im):
                print("Found mask in {}".format(os.path.join(dir_temp, fn)))
                im = process_mask(im)
                if mask is None:
                    mask = im.astype(volume_ref.dtype)
                else:
                    mask += im
    if mask is None:
        mask = np.zeros(volume_ref.shape, dtype=volume_ref.dtype)
        print("WARNING: no masks.")
    if np.any(mask>1):
        print("WARNING: aggregate mask has values > 1")
    
    # Change all positive values to 2 (lesion value).
    mask[mask>0] = 2
    
    # Save aggregate mask.
    mask_f = sitk.GetImageFromArray(mask)
    mask_f.CopyInformation(volume_ref_f)
    if not os.path.exists(os.path.join(dir_target, d_source)):
        os.makedirs(os.path.join(dir_target, d_source))
    sitk.WriteImage(mask_f, os.path.join(dir_target, d_source, case+".nii.gz"))
    
    # Load automatic segmentation, resize if needed.
    case_num = case.split('_')[1]
    mask_auto_fn = "test-segmentation-{}.nii.gz".format(case_num)
    mask_auto_f = sitk.ReadImage(os.path.join(dir_source_auto, mask_auto_fn))
    mask_auto = sitk.GetArrayFromImage(mask_auto_f).astype(volume_ref.dtype)
    mask_auto = process_mask(mask_auto, name="automatic segmentation mask")
    
    # Save automatic segmentation mask.
    mask_auto_f = sitk.GetImageFromArray(mask_auto)
    mask_auto_f.CopyInformation(volume_ref_f)
    if not os.path.exists(os.path.join(dir_target, 'automatic')):
        os.makedirs(os.path.join(dir_target, 'automatic'))
    sitk.WriteImage(mask_auto_f, os.path.join(dir_target,
                                              'automatic', case+".nii.gz"))
                                              
    # Load reference segmentation, resize if needed.
    mask_ref_f = sitk.ReadImage(os.path.join(dir_reference, case+".nii.gz"))
    mask_ref = sitk.GetArrayFromImage(mask_ref_f)
    mask_ref = process_mask(mask_ref, name="reference segmentation mask")
    
    # Save reference segmentation mask.
    mask_ref_f = sitk.GetImageFromArray(mask_ref)
    mask_ref_f.CopyInformation(volume_ref_f)
    if not os.path.exists(os.path.join(dir_target, 'reference')):
        os.makedirs(os.path.join(dir_target, 'reference'))
    sitk.WriteImage(mask_ref_f, os.path.join(dir_target,
                                             'reference', case+".nii.gz"))

    
if not os.path.exists(dir_target):
    os.makedirs(dir_target)
for d_source in dir_source_manual:
    for case in sorted(os.listdir(d_source)):
        if not os.path.isdir(os.path.join(d_source, case)):
            continue
        extract(d_source, case)
        print("")
        print("")
