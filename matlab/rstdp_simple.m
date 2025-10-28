% n1 - the presynaptic neuron. syn is the synapse to be reinforced.
% Plot: top - spike raster. Bottom left - synaptic strength (blue), the
% eligibility trace (green), and the rewards (red x). Bottom right - the
% distribution of synaptic weights with the chosen synapse marked by red dot.
clear;
clc;

M=100;                 % number of synapses per neuron
D=10;                   % maximal conduction delay 
% excitatory neurons   % inhibitory neurons      % total number 
Ne=800;                Ni=200;                   N=Ne+Ni;
a=[0.02*ones(Ne,1);    0.1*ones(Ni,1)];
d=[   8*ones(Ne,1);    2*ones(Ni,1)];
sm=4;                 % maximal synaptic strength

Sn=50; % number of neurons in group
Sg=4; % number of groups %(1 - I1, 2 - I2, 3 - R, 4 - O2)
                         
random_values = randperm(N);

% Reshape the permutation into the desired matrix size
% groups are columns of assigned neurons. 50 rows, elements are neuron
% numbers 0-1000 values
S = reshape(random_values(1:(Sn*Sg)), Sn, Sg);

% rows are all 0-1000 neurons, columns are 100 indices of post neurons
post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)]); 

% initially all excitatory synapses 1 and all inhibitory -1
s=[ones(Ne,M);-ones(Ni,M)];         % synaptic weights
sd=zeros(N,M);                      % their derivatives

% for all N neurons
for i=1:N
  % for all excitatory neurons for all possible conduction delays
  if i<=Ne
    for j=1:D
      % indexes (in post) from neuron i with delay j
      % distribute all post neurons to conduction delays. first 10 to 1,
      % next 10 to 2 ...
      delays{i,j}=M/D*(j-1)+(1:M/D);
    end
  else
    % if inhibitory all conduction delays are 1
    delays{i,1}=1:M;
  end
  % for some neuron find all previous with synaptic weight > 0
  % gives a index running first rows then columns. 
  pre{i}=find(post==i&s>0);             % pre excitatory neurons
  
  % To get neuron 1+mod(pre{i}-1, N)

  %ceil(pre{i}/N) are columns from 100 synapses. /10 gives the cond. delay

  %N*() timestep right. For each fired neuron i find all presynaptic
  %neurons. Not sure why + (D-1), would probably work without as well.
  aux{i}=N*(D-1-ceil(ceil(pre{i}/N)/(M/D)))+1+mod(pre{i}-1,N);
end

% remove non max delay connections to STR
% 
% for i=1:N
%     for del=1:D-1
%         % indices of posts with less than max delay
%         ds = delays{i, del};
%         % for the indices check if they point to STR neuron
%         for k=1:length(ds)
%             % if yes, replace with some other neuron
%             if (ismember(post(i, ds(k)), S(:, 5)))
%                 A = S(:, [1, 2, 3, 4, 6, 7]);
%                 [rows, cols] = size(A);
%                 row_idx = randi(rows);
%                 col_idx = randi(cols);
% 
%                 post(i, ds(k)) = A(row_idx, col_idx);
%             end
%         end
%     end
% end

STDP = zeros(N,3001+D);
v = -65*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
firings=[-D 0];                         % spike timings

%(1 - I1, 2 - I2, 3 - R, 4 - O2)

% set connectons from * to VTA to 0
% us_vta_mask = ismember(post(:,:), S(:, 4));
% s(us_vta_mask) = 0;
% 
% % set connectons from STR to VTA to maximal strength inhibitory
% us_vta_mask = ismember(post(:,:), S(:, 4));
% filter = true(1000, 1);
% filter(S(:, 5), :) = false;
% us_vta_mask(filter, :) = 0;
% s(us_vta_mask) = -sm;
% 
% % set connectons from R to VTA to maximal strength
% us_vta_mask = ismember(post(:,:), S(:, 4));
% filter = true(1000, 1);
% filter(S(:, 3), :) = false;
% us_vta_mask(filter, :) = 0;
% s(us_vta_mask) = sm;

%---------------
% new stuff related to DA-STDP
T=100;         % the duration of experiment
DA=0;           % level of dopamine above the baseline
rew=[];

%shist=zeros(3000*T,2);
%--------------
I=zeros(N,1);
r = 0;
% corr = 0;
for sec=1:T                             % trials
  %r=100-floor(20*rand()); % after cca 100ms
  %paus = 0;
  for t=1:3000                          % simulation of 3 sec WINDOW
    I=zeros(N,1);
    if(sec < 10)
        if mod(t, 300) == 0
            v(S(:, 1))=30;
        elseif mod(t, 300) == 99
            v(S(:, 3))=30;
        end
        
        %r = r+1000-250+floor(500*rand());     % next group firing time 1000+-(0..250)
        %I(S(:, g))=300; %TODO: 30?
        % v(S(:, 1))=30;
    else
        if mod(t, 1000) == 0
            v(S(:, 1))=30;
        end
    end

    % if(corr == 1)
    %     corr = 0;
    %     v(S(:, 3))=30;
    % end

    r = r - 1;
    I=I + 13*(rand(N,1)-0.5);               % random thalamic input

    fired = find(v>=30);                % indices of fired neurons
    v(fired)=-65;  
    u(fired)=u(fired)+d(fired);
    STDP(fired,t+D)=0.1;
    for k=1:length(fired)
      sd(pre{fired(k)})=sd(pre{fired(k)})+STDP(N*t+aux{fired(k)});
    end
    firings=[firings;t*ones(length(fired),1),fired];
    k=size(firings,1);
    while firings(k,1)>t-D
      % neurons from the current fired that shuld fire right now.
      del=delays{firings(k,2),t-firings(k,1)+1};
      % find the post of these that should fire now.
      ind = post(firings(k,2),del);
      % add potential because of fired
      I(ind)=I(ind)+s(firings(k,2), del)';
      % if they already fired as result of something else the connection is
      % much decreased
      sd(firings(k,2),del)=sd(firings(k,2),del)-1.2*STDP(ind,t+D)';
      k=k-1;
    end
    v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical 
    v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
    u=u+a.*(0.2*v-u);                   % step is 0.5 ms
    STDP(:,t+D+1)=0.95*STDP(:,t+D);     % tau = 20 ms
   
    % update actual synapse strength
    DA=DA*0.995;
    if (mod(t,10)==0)
        s(1:Ne,:)=max(0,min(sm,s(1:Ne,:)+(0.002+DA)*sd(1:Ne,:)));
        % s(1:Ne,:)=max(0,min(sm,s(1:Ne,:)+sd(1:Ne,:)));
        sd=0.99*sd;
    end
    
    % reward condition
    rew_delay = 5; %500+ceil(250*rand)
    % is_member_mask = ismember(S(:, 3), fired);
    is_member_mask = [];
    W = 100;
    if size(firings,1) > W
        is_member_mask = ismember(S(:, 3), firings(end-W:end,2));
    end
    sum(is_member_mask(:)) / numel(S(:, 3));
    if (sum(is_member_mask(:)) / numel(S(:, 3)) > 0.1);
    % if (any(ismember(S(:, 3), fired)))
        sec*3000+t+rew_delay
        rew=[rew,sec*3000+t+rew_delay];
        %paus = 1;
    end

    % rewarded input condition
    % is_member_mask = ismember(S(:, 6), fired);

    % Calculate the percentage of elements that are in v
    % percentage_in_v = sum(is_member_mask(:)) / numel(S(:, 6));
    
    % if (percentage_in_v > 0.5)
    %     corr = 1;
    %     r = 1000;
    %     %paus = 1;
    % end
    
    % apply reward if in this moment
    if any(rew==sec*3000+t)
        DA=DA+0.5;
    end

    %shist(sec*3000+t,:)=[s(n1,syn),sd(n1,syn)];

  end
% ---- plot -------
    sec;
  %if(sec > 2500)
      % all neuron raster
      subplot(5,1,1)
      plot(firings(:,1),firings(:,2),'.');
      axis([0 3000 0 N]); 
      
      % I1
      subplot(5,1,2);
      firings_i1 = firings(ismember(firings(:, 2), S(:, 1)), :);
      plot(firings_i1(:,1),firings_i1(:,2),'.');
      axis([0 3000 0 N]);

      % I2 neuron raster
      subplot(5,1,3);
      firings_i2 = firings(ismember(firings(:, 2), S(:, 2)), :);
      plot(firings_i2(:,1),firings_i2(:,2),'.');
      axis([0 3000 0 N]);

      % R neuron raster
      subplot(5,1,4);
      firings_r = firings(ismember(firings(:, 2), S(:, 3)), :);
      plot(firings_r(:,1),firings_r(:,2),'.');
      axis([0 3000 0 N]);

      subplot(5,1,5);
      firings_o2 = firings(ismember(firings(:, 2), S(:, 4)), :);
      plot(firings_o2(:,1),firings_o2(:,2),'.');
      axis([0 3000 0 N]);
      %hist(v,(-100:1:100));
      %fromS2 = s(S(:, 2),:);
      %hist(fromS2(find(fromS2>0)),sm*(0.01:0.01:1)); % only excitatory synapses

      % plotting the action of n1 and synapse after
      %plot(0.001*(1:(sec*1000+t)),shist(1:sec*1000+t,:), 0.001*rew,0*rew,'rx');
      %subplot(2,2,4);
    
      % red dot should move to the right
      %fromS1 = s(S(:, 1),:);
      %hist(fromS1(find(fromS1>0)),sm*(0.01:0.01:1)); % only excitatory synapses
      %hold on; plot(s(n1,syn),0,'r.'); hold off;
    
      drawnow;
  %end
% ---- end plot ------
  STDP(:,1:D+1)=STDP(:,3001:3001+D);
  ind = find(firings(:,1) > 3001-D);
  firings=[-D 0;firings(ind,1)-3000,firings(ind,2)];
  
  %{
  if (sec > 2500 & paus == 1)
    pause(5)
  end
  %}
end
