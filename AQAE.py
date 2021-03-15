# This code is available at https://github.com/randylewis/QuantumAnnealing/AQAE.py
# It uses D-Wave's Ocean Tools, available from https://github.com/dwavesystems/dwave-ocean-sdk
#
# This code can run in quantum annealing mode on D-Wave quantum hardware.
# This code can run in classical simulated annealing mode on your laptop or desktop.
#
# The adaptive quantum annealer eigensolver (AQAE) finds the ground state of a Hamiltonian matrix.
# The AQEA was introduced in our paper: SU(2) gauge theory on a quantum annealer
#                                       A Rahman, Lewis, Mendicelli and Powell
#                                       arXiv:2103.????? [hep-lat] (2021).
# The AQEA algorithm is quite general. See Appendix C of the paper for details.
#
# The particular Hamiltonian coded here is for SU(2) lattice gauge theory on a 6-plaquette lattice.
# This Hamiltonian matrix is equation (B2) in our paper and relates to Figs. 5(a), 6 and 8.
# The specific Hamiltonian matrix can easily be changed in the code below.
# If you find this code useful, please cite the paper as well as the code.

from collections  import defaultdict
from numpy        import sqrt, array, identity, tril_indices, matmul, diag, fill_diagonal, copy
from dwave.system import DWaveSampler, EmbeddingComposite
from neal         import SimulatedAnnealingSampler

# BEGIN USER INPUTS.
quantum = 0     # Choose 0 for SimulatedAnnealingSampler. Choose 1 for DWaveSampler.
myx = 0.2       # Choose the value of the parameter x that appears in the Hamiltonian matrix.
myK = 4         # Choose the initial number of qubits used for each row of the Hamiltonian matrix.
mylambda = -0.3 # Choose the coefficient of the constraint for the QAE algorithm.
myreads = 10000 # Choose the number of samples to be used by the D-Wave sampler.
mychain = 3.0   # Choose a value for the chain penalty.
mytime = 20     # Choose the annealing time in microseconds (integer between 1 and 2000 inclusive).
# END USER INPUTS.

# Define the number of rows in the Hamiltonian matrix.
myB = 13

# Define the Hamiltonian, including the constraint with coefficient lambda.
Hlambda = defaultdict(float)
Hlambda[(1,1)] = 0.0 - mylambda
Hlambda[(1,2)] = -2.0*sqrt(6.0)*myx
Hlambda[(2,2)] = 3.0 - mylambda
Hlambda[(2,3)] = -2.0*myx
Hlambda[(2,4)] = -4.0*myx
Hlambda[(2,5)] = -2.0*sqrt(2.0)*myx
Hlambda[(3,3)] = 4.5 - mylambda
Hlambda[(3,6)] = -2.0*myx
Hlambda[(3,7)] = -2.0*sqrt(2.0)*myx
Hlambda[(4,4)] = 6.0 - mylambda
Hlambda[(4,6)] = -myx/2.0
Hlambda[(4,7)] = -sqrt(2.0)*myx
Hlambda[(4,8)] = -2.0*sqrt(3.0)*myx
Hlambda[(5,5)] = 6.0 - mylambda
Hlambda[(5,7)] = -2.0*myx
Hlambda[(6,6)] = 6.0 - mylambda
Hlambda[(6,9)] = -2.0*myx
Hlambda[(6,10)] = -2.0*myx
Hlambda[(7,7)] = 7.5 - mylambda
Hlambda[(7,9)] = -myx/sqrt(2.0)
Hlambda[(7,10)] = -sqrt(2.0)*myx
Hlambda[(7,11)] = -2.0*myx
Hlambda[(8,8)] = 9.0 - mylambda
Hlambda[(8,10)] = -sqrt(3.0)*myx/2.0
Hlambda[(9,9)] = 7.5 - mylambda
Hlambda[(9,12)] = -2.0*myx
Hlambda[(10,10)] = 9.0 - mylambda
Hlambda[(10,12)] = -myx
Hlambda[(11,11)] = 9.0 - mylambda
Hlambda[(11,12)] = -myx/sqrt(2.0)
Hlambda[(12,12)] = 9.0 - mylambda
Hlambda[(12,13)] = -sqrt(1.5)*myx
Hlambda[(13,13)] = 9.0 - mylambda

# Initialize the center location of the solution vector.
acenter = [0]*myB

# Define the zoom factor (number of extra factors of 2) of the matrix.
zoom = 0

# Repeat the computation with a finer discretization until the user terminates the loop.
while True:

# Define the matrix Q of the AQAE algorithm.
    Q = defaultdict(float)
    for alpha in range(1,myB+1):
        for beta in range(alpha,myB+1):
            if (alpha,beta) in Hlambda:
                for n in range(1,myK+1):
                    i = myK*(alpha-1) + n
                    for m in range(1,myK+1):
                        j = myK*(beta-1) + m
                        if i<=j:
                            if n==myK and m==myK:
                                Q[i,j] = (acenter[alpha-1]-2**(-zoom))*(acenter[beta-1]-2**(-zoom))*Hlambda[alpha,beta]
                            elif n==myK:
                                Q[i,j] = (acenter[alpha-1]-2**(-zoom))*2**(m-myK-zoom)*Hlambda[alpha,beta]
                            elif m==myK:
                                Q[i,j] = (acenter[beta-1]-2**(-zoom))*2**(n-myK-zoom)*Hlambda[alpha,beta]
                            else:
                                Q[i,j] = 2**(n+m-2*myK-2*zoom)*Hlambda[alpha,beta]
                        if i<j:
                            Q[i,j] *= 2

# Send the job to the requested sampler.
    if quantum==1:
        print("Using DWaveSampler")
        sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type__eq':'pegasus'}))
        sampleset = sampler.sample_qubo(Q,num_reads=myreads,chain_strength=mychain,annealing_time=mytime)
        rawoutput = sampleset.aggregate()
    else:
        print("Using SimulatedAnnealingSampler")
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q,num_reads=myreads)
        rawoutput = sampleset.aggregate()

# Record the input choices, display the raw data, and provide headers for the physics output that will follow.
    print("myx=",myx)
    print("myB=",myB)
    print("myK=",myK)
    print("zoom=",zoom)
    print("mylambda=",mylambda)
    print("myreads=",myreads)
    print("mychain=",mychain)
    print("mytime=",mytime)
    print("rawoutput=")
    print(rawoutput)
    print("physics output=")
    if quantum==1:
        print("chain_. ", end =" ")
    print("num_oc.   evalue   evector...")

# Translate the vectors from the Q basis to the H basis, and then display the final results.
    minimumevalue = 100.0
    warning = 0
    chaincount = 0
    for irow in range(len(rawoutput.record)):
        if quantum==1:
            chain = rawoutput.record[irow][3]
        numoc = rawoutput.record[irow][2]
        a = []
        for alphaminus1 in range(myB):
            a.append(0)
            for kminus1 in range(myK-1):
                i = myK*alphaminus1 + kminus1
                a[alphaminus1] += 2**(1+kminus1-myK-zoom)*rawoutput.record[irow][0][i]
            i = myK*alphaminus1 + myK - 1
            a[alphaminus1] += (acenter[alphaminus1]-2**(-zoom))*rawoutput.record[irow][0][i]
        anorm = sqrt(sum(a[i]**2 for i in range(myB)))
        if anorm<1.0e-6:
            print('{:7d}'.format(numoc), "   ---    This vector has length zero.")
            warning += numoc
        else:
            evalue = mylambda + rawoutput.record[irow][1]/anorm**2
            unita = [a[i]/anorm for i in range(myB)]
            if (quantum==1):
                print('{:07.6f}'.format(chain), end =" ")
                if chain>1.0e-6:
                    chaincount += 1
            print('{:7d}'.format(numoc), '{:07.6f}'.format(evalue), ' '.join('{: 07.6f}'.format(f) for f in unita))
            minimumevalue = min(evalue,minimumevalue)
            if evalue==minimumevalue:
                minimuma = a
                minimumunita = unita
                if quantum==1:
                    minimumchain = chain
    print("The minimum evalue from the physics output above is ",'{:07.6f}'.format(minimumevalue))
    print("The normalized evector for the minimum evalue is ",' '.join('{: 07.6f}'.format(f) for f in minimumunita))
    if (quantum==1):
        print("The chain breaking for the minimum evalue is ",'{:07.6f}'.format(minimumchain))
        print("The number of reads that have broken chains is ",chaincount)
    if (warning>0):
        print("WARNING: The number of reads giving the vector of length zero is",warning)

# Construct H_E, H_B, and the full symmetric Hamiltonian matrix without lambda.
    temp_x, temp_y = map(max, zip(*Hlambda)) 
    myH = array([[Hlambda.get((j, i), 0) for i in range(1,temp_y + 1)] for j in range(1,temp_x + 1)])
    myH = myH + mylambda*identity(myB,dtype=float)
    i_lower = tril_indices(myB, -1)
    myH[i_lower] = myH.T[i_lower]
    myHE = diag(diag(myH))
    myHB = copy(myH)
    fill_diagonal(myHB,0.0)

# Calculate <H_E> and <H_B> and <H> for the ground state.
    HEave = matmul(minimumunita,matmul(myHE,minimumunita))
    HBave = matmul(minimumunita,matmul(myHB,minimumunita))
    Have = matmul(minimumunita,matmul(myH,minimumunita))
    print("The normalized evector gives <H_E>, <H_B>, <H> =", HEave, HBave, Have)

# Ask the user whether another computation should be done.
    print()
    print("This has been a computation with myK+zoom =",myK,"+",zoom,"=",myK+zoom)
    query = input("Do you want to try a more precise computation based on this starting point? (yes/no/y/n) ") 
    print("User input=",query)
    if query=="no" or query=="n":
        print()
        print("===============================================================")
        print()
        break

# Prepare for the next computation.
    zoom += 1
    acenter = minimuma
    print()
    print("===============================================================")
    print()
