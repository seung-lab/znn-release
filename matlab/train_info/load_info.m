function [ret] = load_info( fname )
% 
% Load training info
% 
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014

	sInfo = [fname '.info'];
	fInfo = fopen(sInfo, 'r');
	ret.n = fread(fInfo, 1, 'uint64');

	sIter = [fname '.iter'];
	fIter = fopen(sIter, 'r');
	ret.iter = fread(fIter, ret.n, 'uint64');

	sErr = [fname '.err'];
	fErr = fopen(sErr, 'r');
	ret.err = fread(fErr, ret.n, 'double');

	sCls = [fname '.cls'];
	fCls = fopen(sCls, 'r');
	ret.cls = fread(fCls, ret.n, 'double');

end