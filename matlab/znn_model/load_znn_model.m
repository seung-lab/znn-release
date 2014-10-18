function [model] = load_znn_model( fpath )
% 
% Loading saved znn model
% 
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014

	% spec file list
	template = '*.spec';
	specList = dir([fpath '/' template]);

	% load spec
	for i = 1:numel(specList)

		% load spec
		spec_fname = specList(i).name;
		disp(spec_fname);
		[vals{i}] = load_spec( spec_fname );
		[keys{i}] = vals{i}.name;

	end

	% construct model map
	[model] = containers.Map( keys, vals );

	% load weight/biases	
	for i = 1:model.Count

		val = vals{i};
		if isempty(findstr(val.name,'_'))
			% node group
			val.biases = load_bias( val.name, val.size );
		else
			% edge group
			source = model(val.source);
			target = model(val.target);
			val.size = [val.size source.size target.size];
			val.weight = load_weight( val.name, val.size );
		end
		model(val.name) = val;

	end

end