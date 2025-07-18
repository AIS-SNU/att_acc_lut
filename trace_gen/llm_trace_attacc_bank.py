import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16 # FP 16

max_n_hbm = 8
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # byte
n_mac = 16
packing = 1 # packing factor


# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row'] 
HBM_GS['bg']      = n_bank * HBM_GS['ba'] 
HBM_GS['rank']     = n_bg * HBM_GS['bg'] 
HBM_GS['pch']     = n_rank * HBM_GS['rank'] 
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm']


## --------------------------------------  HBM memory space -----------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1   | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []



# n_head and n_req = n_req per a HBM 
def run_gemv(n,k, m , trace_file_name):
  total_cmd = []
  valid_channel = n_channel
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  head_offset = partition_size
  big_offset = pow(2, 20) #

  input_addr_offset = 0
  weight_offset = input_addr_offset + big_offset
  # broadcast input
  ## (pCH) C, C, R, R (MAC)
  ## write input vector to gemv buffer
  # number of partition = (R parallel units)

  # Data broadcasting for pch, rank, bg, and ba
  for ba_idx in range(n_bank): # number of partitions
    for col_idx in range(math.ceil(n/ n_bank / n_mac)):
      for lch in range(math.ceil(valid_channel)):
        # GEMV buffer address, col granularity = 1
        addr = input_addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
        hex_addr = hex(addr)[2:]
        total_cmd.append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))


  # MAC all bank
  ## (pCH) C, C, R, R (MAC)
  # MAC and move output vector to softmax buffer
  ## Vector (1 x k) x Matrix (k x n) multiplication
  ## GEMV unit = adder tree mode
  for n_idx in range(math.ceil( m / n_pch / n_rank / n_bg)):# 16 
    for k_idx in range(math.ceil(k / n_bank / n_mac)): # 2
      idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

      # All bank command (legacy channel)
      for lch in range(math.ceil(valid_channel)):
        addr = weight_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
        total_cmd.append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
        ## parallelization

    ## MVSB command (Move to Softmax buffer) 
    ## A output element is generated for every n_idx
    if n_idx % 16 == 15 or n_idx == math.ceil(m / n_pch / n_rank / n_bg) - 1:
      for bg_idx in range(n_bg):   
        for rank in range(n_rank):
          for lch in range(math.ceil(valid_channel)):
            bank_addr = weight_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                        bg_idx * HBM_GS['bg']
            hex_addr = hex(bank_addr)[2:]
            total_cmd.append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

        
  trace_file = open(trace_file_name, 'w')
  for cmd in total_cmd:
    trace_file.write(cmd + "\n")

  trace_file.close()

def main():
  global dhead, max_L, data_size, n_mac


  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
  parser.add_argument("-n", "--n", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-m", "--m", type=int, default=128,
                      help="Number of heads, default=128")
  parser.add_argument("-k", "--k", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-p", "--packing", type=int, default=1,
                      help="packing_data, default= 1")
  parser.add_argument("-o", "--output", type=str, default="llm_attacc_bank.trace", 
                      help="output path")

  args = parser.parse_args()

  data_size = args.dbyte
  n_mac = int(HBM_GS['col'] / data_size)

  print("------   Make a trace of bank-level AttAcc   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")
  packing_factor = args.packing
  if packing_factor > 1 :
    args.output = args.output.replace(".trace", "_packed.trace")
  run_gemv(args.n, args.m, args.k, packing_factor, args.output)



if __name__ == "__main__":
  main()
