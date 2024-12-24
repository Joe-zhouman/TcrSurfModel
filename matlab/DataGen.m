clc,clear,close all

root = 'H:/Data/1024/test';
dataFilename = [root ,'/Data.csv'];
% surfIdx = 0;
ss = 0;
se = 1999;

E1 = 215e9;
v1=0.26;
E2 = 215e9;
v2=0.26;
E = 1/((1-v1^2)/E1+(1-v2^2)/E2);
% E = 215e9;
H = 2200e6;
k = 0.6;
Ks = 15;
L = 250e-6;
gamma=1.5;%在轮廓中确定频率密度的参数，m
M=10;%用于构建表面重叠脊的数量
Ls=2e-10; %最小截止长度，m
nmax = round(log(L/Ls)/log(gamma));
n1 = round(log(1/L)/log(gamma));
delta = L/1023;
x= delta/2:delta: delta/2 + L;
y= delta/2:delta: delta/2 + L;
[X,Y]=meshgrid(x,y);
sqrtXY = sqrt(X.^2+Y.^2);
atanXY = atan(Y./X);
Aa = L^2;
tic
for surfIdx = ss:se
    D = 1.75+0.2*rand;
    G = 10^(-0.3*rand-10);
    P = 1e6+rand*9e6;
    % 生成表面
    Z = zeros(size(X));
    for m = 1:M
        temp1 = cos(atanXY-pi*m/M);
        for n = 0 : nmax
            phi = rand(size(X))*2*pi;
            Z = Z+(gamma^n)^(D-2)*(cos(phi)-cos(phi+2*pi/L*gamma^n.*sqrtXY*temp1));
        end
    end
    Z = Z * L*(G/L)^(D-1)*sqrt(log(gamma)/M);
    Z = Z-mean(Z,"all");
    filename = ['surf',num2str(surfIdx),'.mat'];
    fullpath = [root,'/Surf/',filename];
    T = rand(size(Z));
    Z1 = Z.*T;
    Z2 = Z-Z1;
    stdZ1 = std(Z1,0,"all");
    stdZ2 = std(Z2,0,"all");
    Z1 = Z1/stdZ1;
    Z2 = Z2/stdZ2;
    save(fullpath,"Z1","Z2",'-mat');
%     save(fullpath2,"Z2",'-mat');
%     clear Z Z1 Z2 T;
    % 计算H

    Fext = P * Aa;
    ac1 = G^2/((k*H/2/E)^(2/(D-1)));
    ac2 = G^2/((3*H/4/E)^(2/(D-1)));
    aL = 1e-10;
    fa =@(a) 16/441 * (3 * (pi/2*(a/ac1).^(1-0.5*D)-1).^2 -...
        2 * (pi/2*(a/ac1).^(1-0.5*D)-1).^3);
    fa2 = @(a) a.^(D/2) .* (1 + fa(a)).*(k*H*2/pi.*(a./ac1).^(0.5-0.25.* D) + ...
        fa(a) .* (H-k*H*2./pi*(a./ac1).^(0.5-0.25.*D)));
    while 1
        if aL < ac2
            F = H*D/(2-D)*aL;
            flag = 1;
        elseif aL > ac1
            temp = quadgk(fa2,ac2,ac1);
            F = H*D/(2-D)*aL^(D/2)*ac2^(1-0.5*D) + D/2*aL^(D/2)*temp + ...
                8*D*E*G/3/pi/(10-D)*aL^(D/2)*(aL^(2.5-0.25*D)-ac1^(2.5-0.25*D));
            flag = 3;
        else
            temp = quadgk(fa2,ac2,aL);
            F = H*D/(2-D)*aL^(D/2)*ac2^(1-0.5*D) + D/2*aL^(D/2)*temp;
            flag = 2;
        end

        f = (F-Fext)/Fext;
        if abs(f) < 1e-4
            break;
        elseif  f > 0
            aL = 0.9*aL;
        else
            aL = 1.1*aL;
        end
    end
    fa3 = @(a) a.^(-D/2).*(1+fa(a));
    switch flag
        case 1
            Ar = D/(2-D)*aL;
        case 2
            Ar = D/(2-D)*ac2 + D/2*aL^(D/2)*quadgk(fa3,ac2,aL);
        case 3
            Ar = D/(2-D)*ac2 + D/2*aL^(D/2)*quadgk(fa3,ac2,ac1) + D/pi/(2-D)*...
                aL^(D/2)*(aL^(1-0.5*D)-ac1^(1-0.5*D));
    end
    Arstar = Ar/Aa;
    fa4 = @(a) sqrt(1 + fa(a)./a.^(D+1));
    if D~=1.5 && flag == 1
        Hc = sqrt(2*aL/pi)*D*Ks/(2-D)/(1-sqrt(Arstar))^1.5;
    elseif D~=1.5 && flag == 2
        Hc = sqrt(2*aL/pi)*D*Ks/(2-D)/(1-sqrt(Arstar))^1.5 * ac2^(0.5-0.5*D)+...
            D*Ks * sqrt(aL^D/pi/(1-sqrt(Arstar))^3)*quadgk(fa4,aL,ac2);
    else
        Hc = sqrt(2*aL/pi)*D*Ks/(2-D)/(1-sqrt(Arstar))^1.5 * ac2^(0.5-0.5*D)+...
            D*Ks * sqrt(aL^D/pi/(1-sqrt(Arstar))^3)*quadgk(fa4,ac1,ac2)+...
            D*Ks*sqrt(aL^D/pi/(1-sqrt(Arstar))^3)*2/(2-D)*(aL^(0.5-0.5*D)-ac1^(0.5-0.5*D));
    end
    fid = fopen(dataFilename,"a+");
    fprintf(fid,"%s,%e,%e,%e,%e,%e,%e,%e\n",filename,G,D,stdZ1,stdZ2,P,Arstar,1e6/Hc);
    fclose(fid);
%     surfIdx = surfIdx + 1;
%     if surfIdx > 1999
%         error("end");
%     end
end

toc