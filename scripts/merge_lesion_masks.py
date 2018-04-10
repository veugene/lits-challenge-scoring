import os
import numpy as np
import SimpleITK as sitk

for d1 in sorted(os.listdir('.')):
    if not os.path.isdir(d1):
        continue
    for d2 in sorted(os.listdir(d1)):
        if not os.path.isdir(os.path.join(d1, d2)):
            continue
        case_num = d2.split('_')[1]
        if case_num not in ['200', '138']:
            continue
        mask = None
        source = None       # Source of image metadata
        print("Processing {} in {}".format(d2, d1))
        for fn in sorted(os.listdir(os.path.join(d1, d2))):
            if not fn.endswith('.nii.gz'):
                continue
            m_f = sitk.ReadImage(os.path.join(d1, d2, fn))
            if source is None:
                source = m_f
            m_np = sitk.GetArrayFromImage(m_f)
            if mask is None:
                mask = m_np
            else:
                if len(mask) != len(m_np):
                    print("WARNING: masks in {} do not have the same number "
                          "of slices. Shortening the longest masks from the "
                          "top.".format(d2))
                if len(mask) < len(m_np):
                    m_np = m_np[-len(mask):]
                if len(mask) > len(m_np):
                    mask = mask[-len(m_np):]
                    source = m_f    # use metadata from shorter image
                mask += m_np
            if np.any(mask>1):
                print("WARNING: mask {} contains values > 1"
                      " -- setting to 1".format(d2))
            mask[mask>0] = 1
        if mask is not None:
            mask_f = sitk.GetImageFromArray(mask)
            mask_f.CopyInformation(source)
            path = os.path.join(d1, "{}.nii.gz".format(d2))
            sitk.WriteImage(mask_f, path)
            print("Written {}".format(path))
