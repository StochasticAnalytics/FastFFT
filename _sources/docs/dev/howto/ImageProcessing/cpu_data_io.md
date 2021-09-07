# CPU data io


## Sample location and files

The code relevant for this How-To can be found in: [cisTEM/src/programs/samples/0_Simple/disk_io_image.cpp](https://github.com/bHimes/cisTEM_downstream_bah/blob/development/src/programs/samples/0_Simple/disk_io_image.cpp)

## Quick and dirty

Images may be read from, and written to disk using a set of "quick and dirty" methods in the ***IMAGE*** class.

```java
int slice_to_read = 1; // read the first slice in Z
Image my_image;
my_image.QuickAndDirtyReadSlice("/path/to/my/image", slice_to_read)
```
see also: [QuickAndDirtyReadSlice](../../reference/classes/Image/QuickAndDirtyReadSlice.md) 
 %        [QuickAndDirtyReadSlices]() 
 %       [QuickAndDirtyWriteSlice]() 
 %        [QuickAndDirtyWriteSlices]()

 ## Image file object

 In order to modify the image header, or to use information about the image without loading it all into memory, you may first create an image file object.
 We most commonly work with MRC files and so there is a specialization of the abstract image file called MRCfile.

 ```java
int slice_to_read = 1; // read the first slice in Z
bool over_write = false;

Image my_image;
MRCFile input_file("/path/to/my/image", over_write)
my_image.ReadSlice(&input_file, 1);  
```
see also: [MRCFile](../../reference/classes/MRCFile/MRCFile.md) 





