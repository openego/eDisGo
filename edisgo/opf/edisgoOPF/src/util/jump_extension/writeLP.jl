using Compat.Printf

if VERSION ≥ v"0.7-"
    const print_shortest = Base.Grisu.print_shortest
    const Printf = Compat.Printf
end
###############################################################################
# Extended writeLP function from JuMP/src/writers.jl
# Extension include: possibility to write quadratic objectives and constraints,
#                    extension to varname_given

###############################################################################
# LP File Writer
# We use the formatting defined at:
#   http://lpsolve.sourceforge.net/5.0/CPLEX-format.htm

varname_generic(m::Model, col::Integer) = "VAR$(col)"

function varname_given(m::Model, col::Integer)
    # TODO: deal with non-ascii characters?
    name = getname(m, col)
    for (pat, sub) in [("[", "_"), ("]", ""),(", ", "_"),(",", "_"),("(",""),(")","")]
        name = replace(name, pat => sub)
    end
    name
end

function writeLP(m::Model, fname::AbstractString; genericnames=true, quadobj=!isempty(m.obj.qvars1))
    varname = genericnames ? varname_generic : varname_given

    f = open(fname, "w")
    if !quadobj
        if length(m.obj.qvars1) != 0
            error("LP writer does not support quadratic objectives.\n")
        end
    end

    # Objective
    if m.objSense == :Max
        write(f,"Maximize\n")
    else
        write(f,"Minimize\n")
    end
    objaff::AffExpr = m.obj.aff
    write(f, " obj: ")
    nnz = length(objaff.coeffs)
    for ind in 1:(nnz-1)
        if ind == 1
            print_shortest(f, objaff.coeffs[ind])
        else
            print_shortest(f, abs(objaff.coeffs[ind]))
        end
        @printf(f, " %s %s ", varname(m, objaff.vars[ind].col), (objaff.coeffs[ind+1] < 0) ? "-" : "+")
    end
    if nnz >= 1
        if nnz == 1
            print_shortest(f, objaff.coeffs[nnz])
        else
            print_shortest(f, abs(objaff.coeffs[nnz]))
        end
        @printf(f, " %s", varname(m, objaff.vars[nnz].col))
    end
    # write quadratic objectives "x^T Q x" in form [x^T Qnew x] / 2 with Qnew = 2*Q
    if quadobj
        qv1 = m.obj.qvars1
        qv2 = m.obj.qvars2
        qc  = m.obj.qcoeffs
        nnz = length(qv1)
        @printf(f," + [ ")
        for ind in 1:(nnz-1)
            if ind == 1
                print_shortest(f, 2*qc[ind])
            else
                print_shortest(f, 2*abs(qc[ind]))
            end
                    
            if qv1[ind] == qv2[ind]
                @printf(f," %s ^2 %s ",varname(m,qv1[ind].col),
                    (qc[ind+1] < 0) ? "-" : "+")
            else
                @printf(f," %s * %s %s ",varname(m,qv1[ind].col),
                    varname(m,qv2[ind].col),(qc[ind+1] < 0) ? "-" : "+")
            end
        end
        if nnz >= 1
            if nnz == 1
                print_shortest(f, 2*qc[nnz])
            else
                print_shortest(f, 2*abs(qc[nnz]))
            end
                    
            if qv1[nnz] == qv2[nnz]
                @printf(f," %s ^2 ",varname(m,qv1[nnz].col))
            else
                @printf(f," %s * %s ",varname(m,qv1[nnz].col),
                    varname(m,qv2[nnz].col))
            end
            # @printf(f," %s %s ",varname(m,c.terms.qvars1[nnz].col),
            #     varname(m,c.terms.qvars2[nnz].col))
        end
        @printf(f,"] / 2")

    end
    @printf(f,"\n")

    # Constraints
    function writeconstrterms(c::LinearConstraint)
        nnz = length(c.terms.coeffs)
        for ind in 1:(nnz-1)
            if ind == 1
                print_shortest(f, c.terms.coeffs[ind])
            else
                print_shortest(f, abs(c.terms.coeffs[ind]))
            end
            @printf(f, " %s %s ", varname(m, c.terms.vars[ind].col), (c.terms.coeffs[ind+1] < 0) ? "-" : "+")
        end
        if nnz >= 1
            if nnz == 1
                print_shortest(f, c.terms.coeffs[nnz])
            else
                print_shortest(f, abs(c.terms.coeffs[nnz]))
            end
            @printf(f, " %s", varname(m, c.terms.vars[nnz].col))
        end
    end

    # Quadratic Constraints
    function writequad(c::QuadConstraint)
        nnz = length(c.terms.qvars1)
        @printf(f,"[ ")
        for ind in 1:(nnz-1)
            if ind == 1
                print_shortest(f, c.terms.qcoeffs[ind])
            else
                print_shortest(f, abs(c.terms.qcoeffs[ind]))
            end
                    
            if c.terms.qvars1[ind] == c.terms.qvars2[ind]
                @printf(f," %s ^2 %s ",varname(m,c.terms.qvars1[ind].col),
                    (c.terms.qcoeffs[ind+1] < 0) ? "-" : "+")
            else
                @printf(f," %s * %s %s ",varname(m,c.terms.qvars1[ind].col),
                    varname(m,c.terms.qvars2[ind].col),(c.terms.qcoeffs[ind+1] < 0) ? "-" : "+")
            end
            # @printf(f," %s %s %s ",varname(m,c.terms.qvars1[ind].col),
            #     varname(m,c.terms.qvars2[ind].col),(c.terms.qcoeffs[ind+1] < 0) ? "-" : "+")
        end
        if nnz >= 1
            if nnz == 1
                print_shortest(f, c.terms.qcoeffs[nnz])
            else
                print_shortest(f, abs(c.terms.qcoeffs[nnz]))
            end
                    
            if c.terms.qvars1[nnz] == c.terms.qvars2[nnz]
                @printf(f," %s ^2 ",varname(m,c.terms.qvars1[nnz].col))
            else
                @printf(f," %s * %s ",varname(m,c.terms.qvars1[nnz].col),
                    varname(m,c.terms.qvars2[nnz].col))
            end
            # @printf(f," %s %s ",varname(m,c.terms.qvars1[nnz].col),
            #     varname(m,c.terms.qvars2[nnz].col))
        end

        @printf(f,"] ")
        for ind in 1:length(c.terms.aff.vars)
            @printf(f,"+ %s %s ",c.terms.aff.coeffs[ind], varname(m,c.terms.aff.vars[ind].col))
        end
        
        if c.sense == :(==)
            @printf(f,"=")
        elseif c.sense == :(<=)
            @printf(f,"<=")
        else
            @printf(f,">=")
        end
        if c.terms.aff.constant != 0
            @printf(f," %s %s ",(c.terms.aff.constant<0) ? "+" : "-", abs(c.terms.aff.constant))
        else
            @printf(f," %s",0)
        end
        @printf(f,"\n")
    end
    write(f,"Subject To\n")
    constrcount = 1
    for i in 1:length(m.linconstr)
        @printf(f, " c%d: ", constrcount)

        c::LinearConstraint = m.linconstr[i]
        rowsense = JuMP.sense(c)
        if rowsense != :range
            writeconstrterms(c)
            if rowsense == :(==)
                @printf(f, " = ")
                print_shortest(f, JuMP.rhs(c))
                println(f)
            elseif rowsense == :<=
                @printf(f, " <= ")
                print_shortest(f, JuMP.rhs(c))
                println(f)
            else
                @assert rowsense == :>=
                @printf(f, " >= ")
                print_shortest(f, JuMP.rhs(c))
                println(f)
            end
            constrcount += 1
        else
            writeconstrterms(c)
            @printf(f, " >= ")
            print_shortest(f, c.lb)
            println(f)
            @printf(f, " c%d: ", constrcount+1)
            writeconstrterms(c)
            @printf(f, " <= ")
            print_shortest(f, c.ub)
            println(f)
            constrcount += 2
        end
    end

    # Quadratic Constraints
    quadcount=1
    for i in 1:length(m.quadconstr)
        @printf(f," qc%d: ", quadcount)
        c::QuadConstraint = m.quadconstr[i]
        writequad(c)
        quadcount += 1
    end

    # SOS constraints
    for i in 1:length(m.sosconstr)
        @printf(f, " c%d: ", constrcount)

        c::SOSConstraint = m.sosconstr[i]
        if c.sostype == :SOS1
            @printf(f, "S1::")
        elseif  c.sostype == :SOS2
            @printf(f, "S2::")
        end

        @assert length(c.terms) == length(c.weights)
        for j in 1:length(c.terms)
            @printf(f, " %s:", varname(m, c.terms[j].col))
            print_shortest(f, c.weights[j])
        end

        println(f)
        constrcount += 1
    end

    # Bounds
    write(f,"Bounds\n")
    for i in 1:m.numCols
        if m.colLower[i] == -Inf
            # No low bound
            if m.colUpper[i] == +Inf
                # Free
                @printf(f, " %s free\n", varname(m, i))
            else
                # x <= finite
                @printf(f, " -inf <= %s <= ", varname(m, i))
                print_shortest(f, m.colUpper[i])
                println(f)
            end
        else
            # Low bound exists
            if m.colUpper[i] == +Inf
                # x >= finite
                @printf(f, " ")
                print_shortest(f, m.colLower[i])
                @printf(f," <= %s <= +inf\n", varname(m, i))
            else
                # finite <= x <= finite
                @printf(f, " ")
                print_shortest(f, m.colLower[i])
                @printf(f, " <= %s <= ", varname(m, i))
                print_shortest(f, m.colUpper[i])
                println(f)
            end
        end
    end

    # Integer - don't handle binaries specially
    write(f,"General\n")
    for i in 1:m.numCols
        t = m.colCat[i]
        (t == :SemiCont || t == :SemiInt) && error("The LP file writer does not currently support semicontinuous or semi-integer variables")
        if t == :Bin || t == :Int
            @printf(f, " %s\n", varname(m, i))
        end
    end

    # Done
    write(f,"End\n")
    close(f)
end
