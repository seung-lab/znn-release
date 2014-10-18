function [ret] = load_spec( fname )
% 
% Loading node & edge group specification
% 
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014

	% read lines from spec file	
	fid = fopen(fname);
	lines = textscan(fid,'%s','Delimiter','\n');
	fclose(fid);

	% line-by-line processing
	lines = lines{1};
	for i = 1:numel(lines)

		% current line
		line = lines{i};

		% name
		if( line(1) == '[' & line(end) == ']' )
			assert(numel(line) > 2);
			assert(~isfield('ret','name'));
			ret.name = line(2:end-1);

			% edge processing
			sepIdx = findstr(ret.name,'_');
			if ~isempty(sepIdx)
				assert(numel(sepIdx) == 1);	% only one separator
				assert(sepIdx > 1);
				assert(numel(ret.name) > sepIdx);
				ret.source = ret.name(1:sepIdx-1);
				ret.target = ret.name(sepIdx+1:end);
			end
		else
			sepIdx = findstr(line,'=');
			assert(~isempty(sepIdx) & (sepIdx > 1));			
			name = line(1:sepIdx-1);
			if( numel(line) > sepIdx )
				value = line(sepIdx+1:end);
			else
				'';
			end
			[ret] = set_name_value_pair(ret,name,value);
		end

	end

end


function [s] = set_name_value_pair( s, name, value )

	switch( name )
	case 'size'
		value = explode_numeric(value,',','%d');
	case {'bias','eta','mom','wc','fft'}
		value = double(str2num(value));
	case {'activation','filter','init_type'}		
	case {'act_params','init_params'}
		value = explode_numeric(value,',','%f');
	case {'filter_size','filter_stride'}
		value = explode_numeric(value,',','%d');
	otherwise		
	end
	[s] = setfield(s,name,value);

end


function [ret] = explode_numeric( str, delim, fmt )

	[delim_pos] = findstr(str,delim);
	switch numel(delim_pos)
	case 0
		ret = str2num(str);
	case 1
		C = textscan(str,[fmt ' ' fmt],'delimiter',delim);
		ret = cell2mat(C);
	case 2
		C = textscan(str,[fmt ' ' fmt ' ' fmt],'delimiter',delim);
		ret = cell2mat(C);
	otherwise
		assert(false);
	end
	ret = double(ret);

end