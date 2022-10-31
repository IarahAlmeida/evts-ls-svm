function my_rllwnn
clc;clear;
close all
%% selecting cases(case(1)=Macky Glass & case(2)=Smoothed_Sunspot_Monthly)
Case=4;
addpath(Path.Data);

global init_sigma sig gama NN

%%
if Case==1
    display('Macky Glass')
    load MG
    

%         init_sigma=0.4
%     gama=2000;
%     sig=0.8;
%     NN=4;
%     Th=7*0.1;
%     Th_merge=7*0.07;

    init_sigma=0.004;
    gama=2800;
    sig=0.9;
    NN=4;
    Th=1*0.00000000000001;
    Th_merge=1*0.000000000000007;
end
if Case==2
    display('competition')
    h=0;
    load data_comp
    n=50; %num of input lags
    Sah=1; % step ahead
    Qte=31;
    Serie=U_Y.Y';
    [train,test]=DATApre(n,Sah,Serie,h,Qte);

    init_sigma=20000;
    gama=2000;
    sig=0.8;
    NN=2;
    Th=1*0.01;
    Th_merge=1*0.007;
end
if Case==3
    display('Smoothed_Sunspot_Monthly')
    load SUN
    init_sigma=270;
    gama=1;
    sig=0.1;
    NN=2;
    Th=      1*0.00000000000001;
    Th_merge=1*0.000000000000007;
%     init_sigma=40000;
%     gama=1;
%     sig=0.1;
%     NN=2;
%     Th=0.5;
% Th_merge=0.47;
end
if Case==4
    Case = 1;
    %init_sigma=0.004;
    %gama=2800;
    %sig=0.0010;
    %Th=0.5;
    %Th_merge=0.56;
     
    init_sigma=0.005;
    gama=700;
    sig=1.0020;
    Th=1*0.00000000000001;
    Th_merge=1*0.000000000000007;
    
    NN=100;
    
    x = loadData(5);  
    train.x = x;
    train.y = train.x(1);
    currentIndex = 2;
    currentSampleSize = size(train.x, 1);
    for t = 2 : currentSampleSize
        first = x(t - 1);
        second = x(t);
        if (first ~= 0 && second ~= 0)
            train.x(currentIndex) = first;
            train.y(currentIndex) = second;
            currentIndex = currentIndex + 1;
        end
    end
    train.y = train.y';
    y = loadData(6);
    test.x = y;
    test.y = test.x(1);
    currentIndex = 2;
    currentSampleSize = size(test.x, 1);
    for t = 2 : currentSampleSize
        first = y(t - 1);
        second = y(t);
        if (first ~= 0 && second ~= 0)
            test.x(currentIndex) = first;
            test.y(currentIndex) = second;
            currentIndex = currentIndex + 1;
        end
    end
    test.y = test.y';
end
if Case == 5
    Case = 1;
    display('CNC')
    load Data4;
    load Data6;
    
    train.x = chkData4(:, 1);
    train.y = chkData4(:, 2);
    
    test.x = chkData6(:, 1);
    test.y = chkData6(:, 2); 

%         init_sigma=0.4
%     gama=2000;
%     sig=0.8;
%     NN=4;
%     Th=7*0.1;
%     Th_merge=7*0.07;

    init_sigma=0.004;
    gama=2800;
    sig=0.9;
    NN=1;
    Th=1*0.00000000000001;
    Th_merge=1*0.000000000000007;
end
%% attantoin:

plotcheck=0;% plotcheck=0(for fast Identification )& plotcheck=1(for slow Identification )

%%

tic;
net=RLLWNN(train,Th,Th_merge,plotcheck);
figure(2)
NRMSE=recall2(net,test,Case);
toc
function net=RLLWNN(train,Th,Th_merge,plotcheck)
%%
clc
close all
num_input=size(train.x,2);
global init_sigma sig gama NN
for kk=1:1
    newcount=0;
    if nargin==4
        net=struct('center',zeros(num_input,1),'sigma',eye(num_input),'invsigma',eye(num_input)...
            ,'num',0,'gama',2000,'sig',0.8,'U',zeros(NN,NN),'x',zeros(NN,4),'y',zeros(NN,1),'phi',zeros(NN+1,NN+1),'P',zeros(NN+1,NN+1),...
            'teta',zeros(NN+1,NN+1),'b',0,'alpha',zeros(NN,1),'y_hat',0);
        net(1).center=train.x(NN,:)';
        net(1).sigma=init_sigma*eye(num_input);
        net(1).invsigma=eye(num_input)/init_sigma;
        net(1).num=1;
        net(1).sig=sig;
        net(1).gama=gama;
        net(1).T=ones(NN,1);
        net(1).x=train.x(1:NN,:);
        net(1).y=train.y(1:NN,:);
        for i=1:NN
            for j=i:NN
                net(1).U(i,j)=exp(-norm(net(1).x(i,:)-net(1).x(j,:))^2/((sig)^2));
            end
        end
        net(1).U=net(1).U+net(1).U'-diag(diag(net(1).U))+eye(NN).*(1/gama);
        net(1).phi=[0,net(1).T';net(1).T,net(1).U];
        net(1).P=inv(net(1).phi);
        net(1).teta=net(1).P*[0;net(1).y];
        net(1).b=net(1).teta(1);
        net(1).alpha=net(1).teta(2:end);
        
        net(1).y_hat=0;
        for h=1:NN
            net(1).y_hat=net(1).y_hat+net(1).alpha(h)*exp(-norm(train.x(NN+1,:)-net(1).x(h,:))^2/(net(1).sig^2));
        end
        
        net(1).y_hat=net(1).y_hat+net(1).b;
        y_e_1=net(1).y_hat;
        y_re=train.y(NN+1);
        error(1)=y_re-y_e_1;
        if plotcheck==1
            subplot(311)
            plot(1,train.y(2),'r')
            hold on
            plot(1,y_e_1,'b')
            hold off
            legend('Real','Estimated')
            drawnow
            subplot(312)
            stem(1,error(1))
            subplot(313)
            plot(1,1,'.-')
            drawnow
        end
        C=1;
        cluster(1)=1;
        newcount=50;
        
    end
    if plotcheck==0
        h=waitbar(0);
    end
    y_e=zeros(size(train.x,1)-1,1);
    y_e(1)=y_e_1;
    error=zeros(size(train.x,1)-1,1);
    error(1)=train.y(NN+1)-y_e_1;
    cluster=zeros(size(train.x,1)-1,1);
    cluster(1)=1;
end

%% ================================ step 2.=============================
for i=NN+1:size(train.x,1)-1
    if i==100 
        pause(1);
    end
    x=train.x(i,:);
    x_next=train.x(i+1,:);
    
    y=train.y(i);
    y_next=train.y(i+1,:);
    %------------------------------------------------------------------
    [net,C]=new_cluster_search(net,x,y,Th,C);
    while C>2
        index=merge_check(net,Th_merge,C,i);
        if index~=0
            net=merge_cluster(net,index);
            C=C-1;
            index(1:2)
        else
            break
        end        
    end
    
    %% alpha_b_update==================================================
    net=alpha_b_update(net,x,y,NN,x_next);
    %------------------------------------------------------------------
    [nSai,Sai,nm,m]=membership(x',net);
    nSai;
    y_e(i-NN+1)=0;
    if ~sum(isnan(nSai))
        for j=1:C
            y_e(i-NN+1)=y_e(i-NN+1)+net(j).y_hat*nSai(j);
        end
    end
    error(i-NN+1)=y_next-y_e(i-NN+1);
    %======================================================
    cluster(i-NN+1)=C;
    
    y_re(i-NN+1,1)=y_next;
    rnmse=sqrt(sum((y_e(1:i-NN+1)-y_re).^2)/sum((y_re-mean(y_re)).^2));
    %% Figures
    if plotcheck==1
        subplot(311)
        plot(1:i-NN+1,y_re,'r')
        hold on
        plot(1:i-NN+1,y_e(1:i-NN+1),'b')
        hold off
        drawnow
        subplot(312)
        plot(1:i-NN+1,error(1:i-NN+1),'.-r')
        drawnow
        subplot(313)
        plot(1:i-NN+1,cluster(1:i-NN+1),'.-k')
        drawnow
    else
        waitbar(i/size(train.x,1),h,mat2str(C))
    end
    
    
end

if plotcheck==0
    close(h)
end

RNMSE=sqrt(sum((y_e(1:end-NN+1)-y_re).^2)/sum((y_re-mean(y_re)).^2));
NMSE=RNMSE^2;
MSE_K=mse(y_e(1:end-NN+1)-y_re);
function [nSai,Sai,nm,m]=membership(data,net)
%%
m=ones(length(net),1);
Sai=ones(length(net),1);
for j=1:length(net)
    u_mu=(data-net(j).center);
    m(j,1)=exp(-0.5*u_mu'*(net(j).invsigma)*u_mu);
    Sai(j,1)=m(j,1);
end
sm=sum(m,1);
if sm==0
    net(1).sigma
    %     pause
end
nm=m/sm;
nSai=(Sai/sm);
function [net,C]=new_cluster_search(net,x,y,Th,C)
%%
global init_sigma sig gama NN
num_input=size(x,2);
[nSai,Sai,nm,m]=membership(x',net);

[dist ind]=max(m,[],1);
dist;
newcluster=dist<Th;

if ~newcluster
    net=clusterupdate(net,x',ind);
end
if newcluster
    newcount=50;
    C=C+1;
    [non nearest]=max(m,[],1);
    net(C).center=x';
    net(C).sigma=init_sigma*eye(num_input);
    net(C).invsigma=eye(num_input)/init_sigma;
    net(C).num=1;
    
    net(C).x=net(nearest).x;
    net(C).y=net(nearest).y;
    
%     [nSai,Sai,nm,m]=membership(x',net);
%     net(C).Y=nSai(C)*net(nearest).y;
    net(C).sig=sig;
    net(C).gama=gama;
    

    net(C).U=net(nearest).U;
    net(C).phi=net(nearest).phi;
    net(C).P=net(nearest).P;
    net(C).teta=net(C).P*[0;net(C).y];
    net(C).b=net(C).teta(1);
    net(C).alpha=net(C).teta(2:end);
end
function index=merge_check(net,Th_merge,C,num)
%%
merge_num=C*(C-1)/2;
count_merge=0;
dist_cluster=zeros(3,merge_num);
for j=1:C
    for k=(j+1):C
        count_merge=count_merge+1;
        D1=exp(-0.5*(net(k).center-net(j).center)'*(net(j).invsigma)*(net(k).center-net(j).center));
        D2=exp(-0.5*(net(j).center-net(k).center)'*(net(k).invsigma)*(net(j).center-net(k).center));
        dist_cluster(:,count_merge)=[j;k;D1];
        if dist_cluster(3,count_merge)>Th_merge & D2>Th_merge
                index=[j,k];
                return
        end
    end
end
index=0;
function net=merge_cluster(prev_net,index)
%%

j=0;
i1=index(1);
i2=index(2);
for i=1:length(prev_net)
    if i~=i1 & i~=i2
        j=j+1;
        net(j)=prev_net(i);
    end
end
net(j+1)=prev_net(1);
num=prev_net(i1).num+prev_net(i2).num;
net(j+1).num=num;
net(j+1).center=(prev_net(i1).num*prev_net(i1).center+prev_net(i2).num*prev_net(i2).center)/num;
net(j+1).sigma=(prev_net(i1).num*prev_net(i1).sigma+prev_net(i2).num*prev_net(i2).sigma)/num+...
    (prev_net(i1).num*prev_net(i2).num)*(prev_net(i1).center-prev_net(i2).center)*(prev_net(i1).center-prev_net(i2).center)'/(num^2);
net(j+1).invsigma=inv(net(j+1).sigma);
net(j+1).P=(prev_net(i1).P);
net(j+1).sig=prev_net(i1).sig;
    net(j+1).gama=prev_net(i1).gama;
    net(j+1).x=prev_net(i1).x;
    net(j+1).y=prev_net(i1).y;
    net(j+1).U=prev_net(i1).U;
    net(j+1).phi=prev_net(i1).phi;
    net(j+1).P=prev_net(i1).P;
    net(j+1).teta=prev_net(i1).P*[0;prev_net(i1).y];
    net(j+1).b=prev_net(i1).teta(1);
    net(j+1).alpha=prev_net(i1).teta(2:end);
function net=clusterupdate(net,x,ind)
%%
beta=net(ind).num/((net(ind).num+1)^2);
gama=net(ind).num/(net(ind).num+1);
net(ind).sigma=gama*net(ind).sigma+beta*(x-net(ind).center)*(x-net(ind).center)';
% net(ind).sigma=((net(ind).num-1)/net(ind).num)*net(ind).sigma+((x-net(ind).center)*(x-net(ind).center)')/(net(ind).num+1);
net(ind).num=net(ind).num+1;
% net(ind).center=(net(ind).num*net(ind).center+x)/(net(ind).num+1);
net(ind).center=net(ind).center+(x-net(ind).center)/(net(ind).num);
function net=alpha_b_update(net,x,y,NN,x_next)
[nSai,Sai,nm,m]=membership(x',net);


for j=1:length(net)
    
    net(j).x=[net(j).x;x];
    net(j).y=[net(j).y;y];
    
    %% step 4.
    
    [r,I]=sort(abs(net(j).alpha.*1./(diag(net(j).P(2:end,2:end)))));
    
    l=I(1);
    net(j).x=[net(j).x(1:l-1,:);net(j).x(l+1:end,:)];
    net(j).y=[net(j).y(1:l-1,:);net(j).y(l+1:end,:)];
    
    P_22=net(j).P(l+1,l+1);
    P_12=[net(j).P(1:l,l+1);net(j).P(l+2:end,l+1)];
    P_21=[net(j).P(l+1,1:l),net(j).P(l+1,l+2:end)];
    P_11=[net(j).P(1:l,1:l),net(j).P(1:l,l+2:end);...
        net(j).P(l+2:end,1:l), net(j).P(l+2:end,l+2:end)];
    if P_22 ~= 0   
        P=P_11-(P_12*P_21)/P_22;
    else % alterei aqui na divisão por 0
        P = P_11-(P_12*P_21);
    end
    net(j).teta=P*[0;net(j).y(1:end-1)];
    %% step 5.
    
    sai(1,1)=1;
    for ii=1:NN-1
        sai(1,ii+1)=exp(-norm(net(j).x(ii,:)-net(j).x(end,:))^2/(net(j).sig^2));
    end
    %---------------------(12b)
    zita=1/(exp(-norm(net(j).x(end,:)-net(j).x(end,:))^2/(net(j).sig^2))+eye(1).*(1/(nSai(j)*net(j).gama)));
    %---------------------(12c)
    nu=1/([sai*P*sai'-1/zita]);
    %---------------------(18)
    z=P*sai';
    %---------------------(19)
    net(j).P=zeros(4,4);
    net(j).P=[ P-nu*z*z'            , nu*z;...
        zita*sai*[nu*z*z'-P] , -nu          ];
    %---------------------
    net(j).e=(net(j).y(end,1)-sai*net(j).teta);
    %-----------------------------------
    net(j).teta=[net(j).teta+nu*z*net(j).e;...
        -nu*net(j).e          ];
    %----------------------------------
    
    net(j).b=net(j).teta(1);
    net(j).alpha=net(j).teta(2:end);
    
    net(j).y_hat=0;
    for h=1:NN
        net(j).y_hat=net(j).y_hat+net(j).alpha(h)*exp(-norm(x_next-net(j).x(h,:))^2/(net(j).sig^2));
    end
    
    net(j).y_hat=net(j).y_hat+net(j).b;
end

%%
function NRMSE_t=recall2(net,test,Case)
%%
global NN
C=length(net);
for i=1:length(test.y)-1
    x=test.x(i,:);
    x_next=test.x(i+1,:);
    
    y=test.y(i);
    y_next=test.y(i+1,:);
    %------------------------------------------------------------------
        
    %% alpha_b_update==================================================
    net=alpha_b_update(net,x,y,NN,x_next);
    %------------------------------------------------------------------
    [nSai,Sai,nm,m]=membership(x',net);
   
    y_e(i)=0;
    for j=1:C
        y_e(i)=y_e(i)+net(j).y_hat*nSai(j);
    end
    error(i)=y_next-y_e(i);
    NRMSE_t=sqrt(sum(error.^2)/sum((test.y-mean(test.y)).^2));
end
if Case==1
    NRMSE_t
elseif Case==3
    NMSE=NRMSE_t^2
elseif Case==2
    MAPE=abs(100*(sum(error./test.y(1:end-1)'))/(length(test.y(1:end-1))))
end
% close all
plot(test.y,'r','LineWidth',2)
hold on
plot(y_e,'-.b','LineWidth',2)
legend('test_r_e_a_l','test_p_r_e_d_i_c_t_e_d')
xlabel('time step')
% set(gca,'fontweight','b')
title('result','fontweight','b')
% grid on;
xlim([0,i])
function [train,test]=DATApre(n,Sah,Serie,h,Qte)
Ndata=length(Serie);
for i=1:n
    Utr(:,i)=Serie(i:end-n+i-Sah,1);
end
if h==1
    Ytr=Serie(Sah+n:end);
    train.x=Utr(1:end-Qte-1,:);
    train.y=Ytr(1:end-Qte-1,1);
    test.x=Utr(end-Qte:end,:);
    test.y=Ytr(end-Qte:end,1);
else
    load data_comp
    for i=1:n
        Utr(:,i)=Serie(i:end-n+i-Sah,1);
    end
    Utr(:,i+1)=(Utr(:,i)-Utr(:,i-1))./Utr(:,i-1);
    for i=1:size(Utr,1)
        Utr(i,n+2:n+5)=U_Y.U(:,n+i)';
    end
    Ytr=Serie(Sah+n:end);
    Qtr=length(Ytr)-Qte;
    test.x=Utr(Qtr+1:end,:);
    test.y=Ytr(Qtr+1:end,1);
    train.x=Utr(1:Qtr,:);
    train.y=Ytr(1:Qtr,1);
end
function x = loadData(cut)      
x = [];
if isscalar(cut)
    name = sprintf('B%04d', cut);
    data = load(name);
    % load capacity as y
    cycleSize = size(data.(name).cycle, 2);
    for i = 1 : cycleSize
       if strcmp(data.(name).cycle(i).type, 'discharge') == 1
           x = [x; data.(name).cycle(i).data.Capacity / 2];
       end
    end
    filename = strcat('capacity_', name, '.mat');
    filedata = x';
    save(filename, 'filedata');
else
    cutCount = 1;
    for currentCut = cut
        name = sprintf('B%04d', currentCut);
        data = load(name);
        % load capacity as y
        cycleSize = size(data.(name).cycle, 2);
        count = 1;
        for i = 1 : cycleSize
           if strcmp(data.(name).cycle(i).type, 'discharge') == 1
               value = data.(name).cycle(i).data.Capacity / 2;
               x(cutCount, count) = value;
               count = count + 1;
           end
        end
        filename = strcat('capacity_', name, '.mat');
        filedata = x(cutCount, :);
        save(filename, 'filedata');
        cutCount = cutCount + 1;
    end
end