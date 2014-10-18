function [ret] = load_bias( name, nBias )
% 
% Loading saved znn node group weight
% 
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014

	fname = [name '.weight'];
	fid = fopen(fname,'r');
	
	% deprecated version
	if fid < 0
		fname = [name '.bias'];
		fid = fopen(fname,'r');	
	end

	ret = zeros(nBias,1);
	ret = fread(fid,nBias,'double');

	fclose(fid);

end