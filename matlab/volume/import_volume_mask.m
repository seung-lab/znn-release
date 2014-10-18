function [vol] = import_volume_mask( fname, dim, ext )
% 
% Import 3D volume mask from file
% 
% Usage:
% 	export_volume( fname )
% 	export_volume( fname, [x y z] )
% 	export_volume( fname, [], ext )
% 	export_volume( fname, [x y z], ext )
% 	
% 	fname:	file name
% 	dim:	3D volume mask dimension
% 			if not exists, read information from [fname.size]
%	ext: 	if exists, file name becomes [fname.ext]
%
% Return:
%	vol		3D volume mask
%
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014
	
	if ~exist('dim','var')
		dim = []
	end

	% volume dimension
	if isempty(dim)
		fsz = fopen([fname '.size'], 'r');
		assert(fsz >= 0);
		x = fread(fsz, 1, 'uint32');
		y = fread(fsz, 1, 'uint32');
		z = fread(fsz, 1, 'uint32');
		dim = [x y z];
	end
	assert(numel(dim) == 3);
	fprintf('dim = [%d %d %d]\n',dim(1),dim(2),dim(3));
	
	% volume
	if exist('ext','var')	
		fvol = fopen([fname '.' ext], 'r');
	else
		fvol = fopen(fname, 'r');
	end	
	assert(fvol >= 0);
	
	vol = false(prod(dim), 1);
	vol = fread(fvol, size(vol), 'uint8');
	vol = logical(reshape(vol, dim));

end