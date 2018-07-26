clear all
close all
delete('diary.txt');
diary('diary.txt')
matfiles = ls('../mat/*.mat');
cellfind = @(string)(@(cell_contents)(strcmp(string,cell_contents)));
% get commit
baseurl = 'https://gitlab.dei.unipd.it/costarob/HDACostaProspero/raw/master/';
options = weboptions('Timeout',Inf);
% jobs.txt
fname_source = 'jobs_tab.txt';
outfname_source = websave(fname_source,strcat(baseurl,'jobs.txt'),options);
file_source = importdata(outfname_source);
jobIDsSource = zeros(length(file_source),1);
code_urls = cell(length(file_source),1);
for i=2:length(file_source)
    tmp = strsplit(file_source{i},',');
    if length(tmp)>1
        jobIDsSource(i) = str2double(tmp{2});
        code_urls{i} = tmp{3};
    end
end
delete(fname_source);
clearvars fname_source outfname_source file_source i
%job_tab.txt
fname_tab = 'job_table_mat.txt';
outfname_tab = websave(fname_tab,strcat(baseurl,'job_table.txt'),options);
file_tab = importdata(outfname_tab);
jobIDs = zeros(length(file_tab),1);
datestrs = cell(length(file_tab),1);
for i=1:length(file_tab)
    tmp = strsplit(file_tab{i},',');
    jobIDs(i) = str2double(tmp{1});
    datestrs{i} = tmp{2};
end
delete(fname_tab);
clearvars fname_tab outfname_tab file_tab i baseurl options

for n=1:size(matfiles,1)
    mat = load(strcat('../mat/',matfiles(n,:)));
    mat.parameters = strrep(mat.parameters,'mean_squared_error','MSE');
    mat.parameters = strrep(mat.parameters,'categorical_cross_entropy','CCE');
    p = strsplit(mat.parameters,'_');
    params = cell(length(p),1);
    arch = strsplit(mat.commento,'_');
    arch = arch{1};
    tmp2=0;
        tmp = strsplit(mat.cl_rep,'\n');
    tmp = tmp(2:end-2);
    tab = zeros(length(tmp),5);
    for i=1:length(tmp)
        s = strsplit(tmp{i});
        s = s(2:end);
        for j=1:size(tab,2)
            tab(i,j) = str2double(s{j});
        end
    end
    if strfind(arch,'FFNN') || 
        continue;
    end
    for i=1:length(p)
        tmp = strsplit(p{i},'-');
        if length(tmp)>1
            params{i+tmp2,1}.name = tmp{1};
            params{i+tmp2,1}.value = tmp{2};
        else
            params{i} = {};
            tmp2 = tmp2-1;
        end
    end
    clearvars tmp i p

    % plot loss and accuracy
    f = figure(1);
    set(f,'Position',[264 200 592 431]);
    plot(1:length(mat.loss),mat.loss);
    hold on;
    plot(1:length(mat.loss),mat.acc);
    plot(1:length(mat.loss),mat.val_loss);
    plot(1:length(mat.loss),mat.val_acc);
    hold off;
    xticks(0:5:length(mat.loss));
    grid on;
%     xtickangle(45);
    xlabel('# epochs');
    ylim([0,1]);
    legend('train loss','train acc','val loss','val acc',...
        'Location','East');
    tit = sprintf("%s T=%s",arch,params{1}.value);
    title(tit);
    
    indTab = find(cellfun(cellfind(mat.datestr),datestrs),1);
    job = jobIDs(indTab);
    if isempty(job)
        continue
    end
    url = code_urls{find(jobIDsSource==job,1)};
    clearvars indTab job tmp2
    disp(tit);
    disp(url);
    url = strrep(url,'Models','main');
    disp(url);
    url = strrep(url,'main','Dataset');
    disp(url);
    url = strrep(url,'Dataset','Utility');
    disp(url);
    tmp = strsplit(url,'/');
    filename = sprintf('%s_%s_%s_%s.png',mat.datestr,arch,mat.parameters,tmp{6});
    saveas(f,filename);
    fprintf('filename: %s',filename);
    printParameters(params)
    disp(mat.cl_rep);

    figure(2);
    subplot(221);
    plot(tab(:,1),tab(:,2));
    title('precision');
    ylim([0,1]);
    subplot(222);
    plot(tab(:,1),tab(:,3));
    title('recall');
    ylim([0,1]);
    subplot(223);
    plot(tab(:,1),tab(:,4));
    title('f1-score');
    ylim([0,1]);
    subplot(224);
    plot(tab(:,1),tab(:,5));
    title('support');
    
%     waitforbuttonpress
    clearvars url tmp tit
end
clearvars n matfiles cellfind f
diary off
function printParameters(p)
    for i=1:size(p,1)
        if ~isempty(p{i})
            fprintf('%10s:%10s\n',p{i}.name,p{i}.value);
        end
    end
end
