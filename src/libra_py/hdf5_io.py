"""One opener for every HDF5 file libra reads or writes: the HDF5 file lock off (the savers are a file's only
writer and re-open it on every saved step, so a concurrent reader holding the default lock aborts the run on the next
save - and a libra reader must not be that reader). h5py is optional at import, required at the first open"""
try:
    import h5py
except ImportError:  # the module imports without h5py - open_hdf5 reports the absence
    h5py = None

_warned_locking = False


def open_hdf5(filename, mode):
    """h5py.File(filename, mode) without the HDF5 file lock - correctness of a reader that races the writer is the
    reader's concern. h5py below 3.5 has no locking argument (TypeError) and h5py over an HDF5 library below 1.12.1
    refuses it (ValueError): the file then opens with the lock and a warning, once, names the environment variable
    that protects the run there"""
    global _warned_locking
    if h5py is None:
        raise ImportError("open_hdf5 needs h5py (pip install h5py)")
    try:
        return h5py.File(filename, mode, locking=False)
    except (TypeError, ValueError) as err:
        if "locking" not in str(err):
            raise
        if not _warned_locking:
            print("WARNING in libra_py.hdf5_io: this h5py / HDF5 pair cannot disable file locking (needs h5py >= 3.5 "
                  "over HDF5 >= 1.12.1) - a concurrent reader holding the lock can abort the run. Export "
                  "HDF5_USE_FILE_LOCKING=FALSE before starting python")
            _warned_locking = True
        return h5py.File(filename, mode)
