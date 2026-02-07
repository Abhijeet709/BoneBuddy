import pydicom
import numpy as np
from pydicom.uid import ImplicitVRLittleEndian

def load_dicom(path_or_file):
    """Load DICOM from a file path (str) or file-like object. Returns (pixel_array, dcm).
    Uses force=True so files with missing or invalid metadata can still be read (pixel data only).
    When Transfer Syntax UID is missing, assumes Implicit VR Little Endian so pixel data can be decoded.
    """
    dicom = pydicom.dcmread(path_or_file, force=True)
    # Allow decoding when file_meta is missing Transfer Syntax UID (common in incomplete DICOMs)
    if getattr(dicom, "file_meta", None) is not None:
        if not getattr(dicom.file_meta, "TransferSyntaxUID", None):
            dicom.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    image = dicom.pixel_array.astype(np.float32)
    return image, dicom
