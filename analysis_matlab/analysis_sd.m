%% SD is defined based on the statistical concept
function [sd] = analysis_sd(A)

[row,clom] = size(A);
% mean
u = sum(sum(A))/(row*clom);
sumSD = 0;
for i=1:row
    for j=1:clom
        sumSD = (A(i,j)-u)^2+sumSD;
    end
end
sd = sqrt(sumSD);
end

