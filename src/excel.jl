using CSV
using Tables

"""
    xlclip()

Copy Excel-copied data from the clipboard into a Julia vector or matrix (depending on shape). Single column or row are converted to Julia `Vector`s.
"""
function xlclip()
    xlclip_reader(InteractiveUtils.clipboard())
end

# use a barrier function to allow precompilation/testing of expensive part without 
# messing with user's clipboard
function xlclip_reader(str)
    res = CSV.File(IOBuffer(str),header=false) |> Tables.matrix
    m, n = size(res)
    if m == 1 || n == 1
        return vec(res)
    else
        return res
    end
end

"""
    xlclip()

Copy Julia array to the clipboard in an Excel-friendly format.

Vectors will be copied as Excel columns; to copy a vector `v` to a row for Excel, you can transpose it: `xlclip(v')`
"""
function xlclip(data)
    InteractiveUtils.clipboard(xlclip_writer(data)) # drop the trailing newline
end

function xlclip_writer(data)
    #Tables.table wants an array, not vector so convert if vector
    if typeof(data) <: Vector
        data = reshape(data, :,1)
    end
    rows = collect(CSV.RowWriter(Tables.table(data),delim = "\t"))
    result = join(rows[2:end]) # don't include the column names
   return result[1:end-1] # drop the trailing newline
end