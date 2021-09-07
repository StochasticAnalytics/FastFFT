# MRCFile

Inherits from: AbstractImageFile

## Constructor

Has copy constructor: true

```c++
MRCFile();
MRCFile(std::string filename, bool overwrite = false);
MRCFile(std::string filename, bool overwrite, bool wait_for_file_to_exist);
```

## Properties

### public

std::fstream *my_file;
MRCHeader my_header;
wxString filename;
bool do_nothing; // under special circumstances (like when the file is /dev/null), let's not do anything at all
bool rewrite_header_on_close;
int max_number_of_seconds_to_wait_for_file_to_exist;

```{note}
Arguably all of these properties should probably be private. 
```

### private

```{sidebar} TODO
search code to see if these are infact treated as private, and if so modify them there.
```

## Methods


inline int ReturnXSize() {MyDebugAssertTrue(my_file->is_open(), "File not open!");	return my_header.ReturnDimensionX();};
inline int ReturnYSize() {MyDebugAssertTrue(my_file->is_open(), "File not open!");	return my_header.ReturnDimensionY();};
inline int ReturnZSize() {MyDebugAssertTrue(my_file->is_open(), "File not open!");	return my_header.ReturnDimensionZ();};
inline int ReturnNumberOfSlices() {MyDebugAssertTrue(my_file->is_open(), "File not open!");	return my_header.ReturnDimensionZ();};

inline bool IsOpen() {return my_file->is_open();}

[OpenFile](OpenFile.md)
void CloseFile();
void FlushFile();

% I think this is typicall only called from image methods ReadSlice etc
[ReadSliceFromDisk](ReadSliceFromDisk.md) 
void ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array);

inline void WriteSliceToDisk(int slice_number, float *input_array) {WriteSlicesToDisk(slice_number, slice_number, input_array);}
void WriteSlicesToDisk(int start_slice, int end_slice, float *input_array);

inline void WriteHeader() {my_header.WriteHeader(my_file);};

void PrintInfo();

float ReturnPixelSize();
void SetPixelSize(float wanted_pixel_size);
inline void SetDensityStatistics( float wanted_min, float wanted_max, float wanted_mean, float wanted_rms ){my_header.SetDensityStatistics(wanted_min, wanted_max, wanted_mean, wanted_rms);}