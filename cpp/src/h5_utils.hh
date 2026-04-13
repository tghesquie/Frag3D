#pragma once
#include <filesystem>
#include <hdf5.h>
#include <string>

namespace h5util {

// open existing or create new file
inline hid_t open_or_create_file(const std::string &path) {

  hid_t fid;
  if (std::filesystem::exists(path)) {
    fid = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  } else {
    fid = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }
  return fid;
}

// (re)create a group; if already exists, remove it first
inline hid_t recreate_group(hid_t fid, const std::string &name) {
  if (H5Lexists(fid, name.c_str(), H5P_DEFAULT) > 0)
    H5Ldelete(fid, name.c_str(), H5P_DEFAULT);
  return H5Gcreate2(fid, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

// scalar attributes
inline void write_attr_double(hid_t gid, const char *key, double value) {
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_NATIVE_DOUBLE);
  hid_t attr = H5Acreate2(gid, key, atype, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, atype, &value);
  H5Aclose(attr);
  H5Tclose(atype);
  H5Sclose(space);
}

inline void write_attr_int(hid_t gid, const char *key, int value) {
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_NATIVE_INT);
  hid_t attr = H5Acreate2(gid, key, atype, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, atype, &value);
  H5Aclose(attr);
  H5Tclose(atype);
  H5Sclose(space);
}

// 1D dataset of doubles
inline void write_dataset_1d(hid_t gid, const char *name, const double *data,
                             hsize_t n) {
  hsize_t dims[1] = {n};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(gid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(dset);
  H5Sclose(space);
}

// 2D dataset of doubles (row-major: n0 rows, n1 cols)
inline void write_dataset_2d(hid_t gid, const char *name, const double *data,
                             hsize_t n0, hsize_t n1) {
  hsize_t dims[2] = {n0, n1};
  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t dset = H5Dcreate2(gid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(dset);
  H5Sclose(space);
}

} // namespace h5util
