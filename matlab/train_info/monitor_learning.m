function [data] = monitor_learning( cost_type, avg_winidow, start_iter )

	%% Options
	%
	if ~exist('cost_type','var')
		cost_type = 'Cost';
	end	
	if ~exist('avg_winidow','var')
		avg_winidow = 0;
	end
	if ~exist('start_iter','var')
		start_iter = [1 1];
	end

	% Load train info
	[train] = load_info('train');

	% Load test info
	[test] = load_info('test');

	% [kisuklee] TEMP
	idx = (train.iter == 0);
	train.iter(idx) = [];
	train.err(idx) 	= [];
	train.cls(idx) 	= [];
	idx = (test.iter == 0);
	test.iter(idx) 	= [];
	test.err(idx) 	= [];
	test.cls(idx) 	= [];

	if strcmp(cost_type,'RMSE')
		train.err = sqrt(train.err);
		test.err = sqrt(test.err);
	end

	% windowing
	train.iter 	= train.iter(start_iter(1):end);
	train.err 	= train.err(start_iter(1):end);
	train.cls 	= train.cls(start_iter(1):end);
	train.n 	= numel(train.iter);
	test.iter 	= test.iter(start_iter(2):end);
	test.err 	= test.err(start_iter(2):end);
	test.cls 	= test.cls(start_iter(2):end);
	test.n 		= numel(test.iter);

	% convolution filter
	[train] = smooth_curve( train, avg_winidow );
	[test] 	= smooth_curve( test, avg_winidow );
	if( avg_winidow > 0 )
		avgStr = [', smoothing window = ' num2str(avg_winidow)];
	else
		avgStr = '';
	end

	% return data
	data.train = train;
	data.test = test;
	data.cost = cost_type;

	% Plot cost
	figure;
	hold on;
	grid on;

		h1 = plot(train.iter, train.err, '-k');
		h2 = plot(test.iter, test.err, '-r');
					
		% axis([0 max(train.iter) 0 max(train.err)]);
		xlabel('iteration');
		ylabel(cost_type);
		title(['cost' avgStr]);
		legend([h1 h2],'train','test');

	hold off;

	% Plot classification error
	figure;
	hold on;
	grid on;

		h1 = plot(train.iter, train.cls, '-k');
		h2 = plot(test.iter, test.cls, '-r');

		% axis([0 max(train.iter) 0 max(train.err)]);
		xlabel('iteration');
		ylabel('classification error');
		title(['classification error' avgStr]);
		legend([h1 h2],'train','test');

	hold off;

end
