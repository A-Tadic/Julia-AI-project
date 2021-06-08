#First what I'll do is program this up using onyl Zygote, then I'll do it all with Flux

using Zygote
using Flux

ws = randn(6)
bs = randn(3)



sigmoid(x) = 1/(1+exp(-x))

neuron(x1,w1,x2,w2,b) = sigmoid(b + x1*w1 + x2*w2)

xorNN(x1,x2,w11,w12,w21,w22,w31,w32,b1,b2,b3) =
let 
	n1 = neuron(x1,w11,x2,w12,b1) 
	n2 = neuron(x1,w21,x2,w22,b2)
	n3 = neuron(n1,w31,n2,w32,b3)
return n3
end

error(W,b) = 
let 
	dxor10 = (1.0 - xorNN(1.0,0.0,W[1],W[2],W[3],W[4],W[5],W[6],b[1],b[2],b[3]))^2
	dxor01 = (1.0 - xorNN(0.0,1.0,W[1],W[2],W[3],W[4],W[5],W[6],b[1],b[2],b[3]))^2
	dxor00 = (0.0 - xorNN(0.0,0.0,W[1],W[2],W[3],W[4],W[5],W[6],b[1],b[2],b[3]))^2
	dxor11 = (0.0 - xorNN(1.0,1.0,W[1],W[2],W[3],W[4],W[5],W[6],b[1],b[2],b[3]))^2
return dxor10 + dxor01 + dxor00 + dxor11
end

e = error(ws, bs)
println(e)
println(" ")

n = 0

while n < 100 && e > 0
	gs = gradient(() -> error(ws, bs), params(ws, bs))
	#for some reason, replacing ws, bs with [ws, bs] screws things up here
	#It still lets you take gs[ws] and gs[bs] for some reason

	newws = gs[ws]
	newbs = gs[bs]
	
	#Can also do this instead of the above
	#gs = gradient(error,ws,bs)

	#newws = gs[1]
	#newbs = gs[2]
	
	ws .-= 2 .* newws
	bs .-= 2 .* newbs
	
	global e = error(ws, bs)
	
	global n+=1
end



println(e)
println(" ")


#println(gs[ws])
#println(" ")
#println(error(ws, bs))
#println(" ")
println(xorNN(1,1,ws[1],ws[2],ws[3],ws[4],ws[5],ws[6],bs[1],bs[2],bs[3]))

println(xorNN(1,0,ws[1],ws[2],ws[3],ws[4],ws[5],ws[6],bs[1],bs[2],bs[3]))
println(xorNN(0,1,ws[1],ws[2],ws[3],ws[4],ws[5],ws[6],bs[1],bs[2],bs[3]))
println(xorNN(0,0,ws[1],ws[2],ws[3],ws[4],ws[5],ws[6],bs[1],bs[2],bs[3]))