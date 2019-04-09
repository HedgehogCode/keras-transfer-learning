/*
 * Macro to create a segmentation dataset from generated images.
 */

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "Images suffix", value = "low-noise") suffix
#@ Integer (label = "Train size", value = "800") numTrain

in_img_dir = input + File.separator + "images_" + suffix;
in_label_dir = input + File.separator + "labels";

train_dir = output + File.separator + "train" + File.separator;
train_img_dir = train_dir + "images" + File.separator;
train_labels_dir = train_dir + "masks" + File.separator;

test_dir = output + File.separator + "test" + File.separator;
test_img_dir = test_dir + "images" + File.separator;
test_labels_dir = test_dir + "masks" + File.separator;

File.makeDirectory(train_dir)
File.makeDirectory(train_img_dir)
File.makeDirectory(train_labels_dir)

File.makeDirectory(test_dir)
File.makeDirectory(test_img_dir)
File.makeDirectory(test_labels_dir)

processImages(in_img_dir)
processLabels(in_label_dir)

function processImages(in_img_dir) {
	list = getFileList(in_img_dir);
	list = Array.sort(list);
	// Train images
	for (i = 0; i < numTrain; i++) {
		loadAndSave(in_img_dir + File.separator + list[i], train_img_dir, i);
	}
	// Test images
	for (i = 0; i < list.length - numTrain; i++) {
		loadAndSave(in_img_dir + File.separator + list[i + numTrain], test_img_dir, i);
	}
}

function processLabels(in_label_dir) {
	list = getFileList(in_label_dir);
	list = Array.sort(list);
	// Train images
	for (i = 0; i < numTrain; i++) {
		loadReduceAndSave(in_label_dir + File.separator + list[i], train_labels_dir, i);
	}
	// Test images
	for (i = 0; i < list.length - numTrain; i++) {
		loadReduceAndSave(in_label_dir + File.separator + list[i + numTrain], test_labels_dir, i);
	}
}

function loadAndSave(file, output_dir, index) {
	if(File.isDirectory(file)) {
		print("Ignoring directory " + list[i])
	} else if(endsWith(file, ".tif")) {
		print("Processing image " + file);
		open(file);
		output_file = output_dir + format(index) + ".tif";
		print("Saving image " + output_file);
		saveAs("Tiff", output_file);
		close();
	}
}

function loadReduceAndSave(file, output_dir, index) {
	if(File.isDirectory(file)) {
		print("Ignoring directory " + list[i])
	} else if(endsWith(file, ".tif")) {
		print("Processing labeling " + file);
		open(file);
		run("Z Project...", "projection=[Max Intensity]");
		output_file = output_dir + format(index) + ".tif";
		print("Saving labeling " + output_file);
		saveAs("Tiff", output_file);
		close();
		close();
	}
}

function format(number) {
	if (number < 10) {
		return "0000" + number;
	} else if ( number < 100) {
		return "000" + number;
	} else if ( number < 1000) {
		return "00" + number;
	} else if ( number < 10000) {
		return "0" + number;
	} else if ( number < 100000) {
		return "" + number;
	} else {
		print("ERROR: Number too large")
	}
}
