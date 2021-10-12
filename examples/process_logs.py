import os
path = "./logs" 
# path = "./yard_logs" 
files= os.listdir(path) 
s = []

model_name = ""
gpu_num = 0
bs = 0

res_dict = {}
file_dict = {}
for file in files:
    # log.GPT2_2B_bs_8_cs_48_gpu_4_cpueb_1
    if not os.path.isdir(file):
        fn_list = file.split('.')[1].split('_')
        for i in range(len(fn_list)):
            if 'GPT' in fn_list[i]:
                model_name = fn_list[i+1]
            elif 'bs' == fn_list[i]:
                bs = fn_list[i+1]
            elif 'gpu' == fn_list[i]:
                gpu_num = fn_list[i+1]
        key = model_name + "_" + bs + "_" + gpu_num
        f = open(path+"/"+file); 
        iter_f = iter(f); 
        for line in iter_f: 
            if 'Tflops' in line and 'WARM' not in line:
                sline = line.split()
                perf = float(sline[-2])
                if key not in res_dict:
                    res_dict[key] = perf
                    file_dict[key] = file
                else:
    		    if res_dict[key] < perf:
		        res_dict[key] = perf 
                        file_dict[key] = file

print(res_dict)
print(file_dict)
