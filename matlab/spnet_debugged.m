% n1 - the presynaptic neuron. syn is the synapse to be reinforced.
% Plot: top - spike raster. Bottom left - synaptic strength (blue), the
% eligibility trace (green), and the rewards (red x). Bottom right - the
% distribution of synaptic weights with the chosen synapse marked by red dot.
rand('seed',1);
M=6;%100;                 % number of synapses per neuron
D=3;                   % maximal conduction delay 
% excitatory neurons          
Ne=8;%800;  
% inhibitory neurons
Ni=2;%200;

%{
D=2;
M=1;
Ne=4;
Ni=0;
%}

% total number
N=Ne+Ni;
a=[0.02*ones(Ne,1);    0.1*ones(Ni,1)];
d=[   8*ones(Ne,1);    2*ones(Ni,1)];
sm=100;                 % maximal synaptic strength


post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)])
s=[60*ones(Ne,M);-50*ones(Ni,M)]         % synaptic weights


post=[2;4;1;2];
s=[300;0;300;0];
delays=cell(4,4);
delays{1,2}=1;
delays{2,1}=1;
delays{3,1}=1;
delays{4,1}=1;

sd=zeros(N,M);                      % their derivatives

for i=1:N
  %{
  if i<=Ne
    for j=1:D
      delays{i,j}=M/D*(j-1)+(1:M/D);
    end;
  else
    delays{i,1}=1:M;
  end;
  %}
  
  pre{i}=find(post==i&s>0);             % pre excitatory neurons
  aux{i}=N*(D-1-ceil(ceil(pre{i}/N)/(M/D)))+1+mod(pre{i}-1,N);
end;
delays
post
s

STDP = zeros(N,11+D);
v = -65*ones(N,1);                      % initial values
u = 0.2.*v;                             % initial values
firings=[-D 0];                         % spike timings

T=1;%3600;         % the duration of experiment
I = zeros(N,1);

%I(1)=300;
for sec=1:T                             % simulation of 1 day
  for t=1:10                          % simulation of 1 sec
      t
      
    if t==1
        I(1)=300;
    elseif t==4
        I(3)=300;
    end;
      
      %I=13*(rand(N,1)-0.5);               % random thalamic input 
    fired = find(v>=30);                % indices of fired neurons
    v(fired)=-65;  
    u(fired)=u(fired)+d(fired);
    STDP(fired,t+D)=0.1;
    
    for k=1:length(fired)
      sd(pre{fired(k)})=sd(pre{fired(k)})+STDP(N*t+aux{fired(k)});
    end;
    sd
    
    firings=[firings;t*ones(length(fired),1),fired]
    k=size(firings,1);
    while firings(k,1)>t-D
      del=delays{firings(k,2),t-firings(k,1)+1}; % seq. indexes! of neighbors with t-f(..) delay
      ind = post(firings(k,2),del) % from seq to actual neuron (get del-th neighbor from f(k, 2)th. neuron)
      firings(k,2)
      I(ind)=I(ind)+s(firings(k,2), del)'; %update neighbors in the future
      sd(firings(k,2),del)=sd(firings(k,2),del)-1.5*STDP(ind,t+D)'; %pre-post synapses are less 
      k=k-1;
    end;
    sd
    v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical 
    v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time 
    u=u+a.*(0.2*v-u);                   % step is 0.5 ms
    STDP(:,t+D+1)=0.95*STDP(:,t+D)     % tau = 20 ms
    I = zeros(N,1);
  end;
% ---- plot -------

  plot(firings(:,1),firings(:,2),'.');
  axis([0 10 0 N]); drawnow;


  STDP(:,1:D+1)=STDP(:,11:11+D);
  ind = find(firings(:,1) > 11-D);
  firings=[-D 0;firings(ind,1)-1000,firings(ind,2)];
  s
  s(1:Ne,:)=max(0,min(sm,0.01+s(1:Ne,:)+sd(1:Ne,:)))
  sd=0.9*sd;

end;
