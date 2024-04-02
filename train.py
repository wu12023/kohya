import os
import subprocess
import shutil
import multiprocessing
def list_subdirectories(directory):
    subdirectories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            subdirectories.append(item)
    return subdirectories

def copy_images(src_dir, dest_dir, num_images=85):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i, file_name in enumerate(os.listdir(src_dir)):
        if i >= num_images:  # 限制为前85张图片
            break
        full_file_name = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)

def process_task(subdir, gpu_id, log_file):
    print(f'start training {subdir} with gpu{gpu_id}!!!')
    src_folder = os.path.join(lora_model_dir, subdir)
    image_folder = os.path.join("/root/shiym_proj/kohya_ss/dataset", subdir, tag)
    copy_images(src_folder, image_folder)
    dest_folder = os.path.join("/root/shiym_proj/kohya_ss/dataset", subdir)
    command = f"accelerate launch --gpu_ids='{gpu_id}' --num_cpu_threads_per_process=8 '/root/shiym_proj/kohya_ss/sd-scripts/sdxl_train_network.py' --train_data_dir='{dest_folder}' --output_dir='/root/shiym_proj/kohya_ss/outputs/{subdir}' --output_name='{subdir}' --bucket_no_upscale --bucket_reso_steps=32 --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --gradient_checkpointing --learning_rate='1.0' --lr_scheduler='constant' --lr_scheduler_num_cycles='5' --max_data_loader_n_workers='0' --max_grad_norm='1' --resolution='1024,1344' --max_train_epochs=5 --max_train_steps='5000' --min_snr_gamma=5 --mixed_precision='bf16' --network_alpha='4' --network_args 'preset=full' 'conv_dim=4' 'conv_alpha=4' 'rank_dropout=0' 'module_dropout=0' 'factor=-1' 'use_cp=False' 'use_scalar=False' 'decompose_both=False' 'rank_dropout_scale=False' 'algo=lokr' 'train_norm=False' --network_dim=4 --network_module=lycoris.kohya --network_train_unet_only --no_half_vae --noise_offset=0.0357 --optimizer_args d_coef=1.0 weight_decay=0.01 safeguard_warmup=False use_bias_correction=False  --optimizer_type='Prodigy' --pretrained_model_name_or_path='/root/shiym_proj/diffusers/examples/dreambooth/stable-diffusion-xl-base-1.0' --save_model_as=safetensors --save_precision='float' --train_batch_size='4'  --unet_lr=1.0 --xformers --sample_sampler=euler_a --sample_prompts='/root/kohya_ss/outputs/only_attn1_without_caption/sample/prompt.txt' --sample_every_n_epochs=10000 --sample_every_n_steps=10000"
    subprocess.run(command, shell=True)
    print(f'finish training {subdir} with gpu{gpu_id}!!!')
    append_log(log_file, f"Finished processing {subdir} on GPU {gpu_id}")

def append_log(log_file, message):
    with open(log_file, 'a') as file:
        file.write(message + '\n')
def validate_train(log_file,subdir):
    with open(log_file,'r') as file:
        validate = file.read()
        if subdir in validate:
            return False
        else:
            return True
# 读取所有文件夹
lora_model_dir = "/root/shiym_proj/stable-diffusion-webui/api_out/txt2img"
subdirectories = list_subdirectories(lora_model_dir)

# tag 定义
tag = "5_sks girl"

# 日志文件路径
log_file = "/root/shiym_proj/kohya_ss/trainlog.txt"

# 使用 multiprocessing 创建进程池
pool = multiprocessing.Pool(1)  # 假设你有两个 GPU

# 分配任务到进程
for i, subdir in enumerate(subdirectories):
    if validate_train(log_file,subdir):
        gpu_id = 0 # 这会交替地分配 GPU 0 和 GPU 1
        pool.apply_async(process_task, args=(subdir, gpu_id, log_file))
    else:
        print("finished before,skip")

# 关闭进程池并等待所有进程完成
pool.close()
pool.join()

print("All tasks completed. Log written to", log_file)