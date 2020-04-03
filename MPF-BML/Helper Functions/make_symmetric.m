function out = make_symmetric(J)

total_length = sqrt(length(J));
Jmat = reshape(J,total_length,total_length);
Jmat = (Jmat+Jmat')/2;
out = Jmat(:);