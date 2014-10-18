function [ret] = load_weight( name, sz )
% 
% Loading saved znn edge group weight
% 
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014

	fname = [name '.weight'];
	fid = fopen(fname,'r');
	
	nWeights = prod(sz);
	ret = zeros(nWeights,1);
	ret = fread(fid,nWeights,'double');

	source = sz(4);
	target = sz(5);
	sz(4) = target;
	sz(5) = source;
	ret = reshape(ret,sz);
	ret = permute(ret,[1 2 3 5 4]);

	fclose(fid);

end