/**
 * Copyright (2019) Yundea IOT Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * File: recorder.c
 * Auth: Jim meng (alongmh@163.com)
 * Desc: Record module function implementation.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <alsa/asoundlib.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define ALSA_PCM_NEW_HW_PARAMS_API
#define SAMPLE_RATE         			(16000)
#define FRAMES_INIT         			(640*4)
#define CHANNEL 	 	  			(4)
#define FRAMES_SIZE  	  			((16/8) *CHANNEL)
#define PCM_STREAM_CAPTURE_DEVICE	"hw:2,0"
//#define PCM_STREAM_CAPTURE_DEVICE	"default"

#define MYPORT  8877
#define QUEUE 20
#define BUFFER_SIZE 1024

// #define PRINTF printf
#define PRINTF

typedef struct{
    int dir;
    int size;
    unsigned int val;
    snd_pcm_t *handle;
    snd_pcm_uframes_t frames;
    snd_pcm_hw_params_t *params;
}rec_config_t;

static rec_config_t *s_index = NULL;
	

typedef struct _pcm_header_t {
    char    pcm_header[4];
    uint32_t  pcm_length;
    char    format[8];
    int     bit_rate;
    short   pcm;
    short   channel;
    int     sample_rate;
    int     byte_rate;
    short   block_align;
    short   bits_per_sample;
    char    fix_data[4];
    uint32_t  data_length;
} pcm_header_t;

static pcm_header_t s_pcm_header = {
    {'R', 'I', 'F', 'F'},
    ( uint32_t)-1,
    {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '},
    0x10,
    0x01,
    0x01,
#if 1 //def SAMPLE_RATE_16K
    0x3E80,
    0x7D00,
#else //8k
    0x1F40,
    0x3E80,
#endif // SAMPLE_RATE_16K
    0x02,
    0x10,
    {'d', 'a', 't', 'a'},
    (uint32_t)-1
};


typedef struct _REC_FILE{
	FILE    *_file;
	pcm_header_t _hdr;
}REC_FILE;

REC_FILE  *s_rec_files[CHANNEL+1];


// 一个wav文件分为两个部分
// 1. s_pcm_header
// 2. mono_buffer
typedef struct{
    pcm_header_t _hdr;
    int16_t *mono_buffer;
    uint32_t mono_size;
}wav_buffer;
// wav_buffer  *wav_buffers[CHANNEL];

int sock_cli;

typedef struct dataPackage{
    uint32_t size0;
    uint32_t size1;
    uint32_t size2;
    uint32_t size3;
    short *data0;
    short *data1;
    short *data2;
    short *data3;
}dataPackage;

REC_FILE * duer_store_voice_start(int channel_id)
{
	REC_FILE *file=NULL;
        
    PRINTF("start");
    file = (REC_FILE*)malloc(sizeof(REC_FILE));
    if(file==NULL){
        return NULL;	
	}
	
    char _name[64];
    snprintf(_name, sizeof(_name), "./channel-%d.wav", channel_id);
    file->_file = fopen(_name, "wb");
    if (!file->_file ) {
        PRINTF("can't open file %s", _name);
        return  NULL;
    } else {
        PRINTF("begin write to file:%s", _name);
    }

    
    memcpy(&file->_hdr,&s_pcm_header,sizeof(s_pcm_header));
    fwrite(&s_pcm_header, 1, sizeof(s_pcm_header), file->_file);
    file->_hdr.data_length = 0;
    file->_hdr.pcm_length = sizeof(s_pcm_header) - 8;

    // 初始化
    // wav_buffers[channel_id] = (wav_buffer*)malloc(sizeof(wav_buffer));
    // wav_buffers[channel_id]->mono_size = 0;

    // memcpy(&wav_buffers[channel_id]->_hdr,&s_pcm_header,sizeof(s_pcm_header));
    // wav_buffers[channel_id]->_hdr.data_length = 0;
    // wav_buffers[channel_id]->_hdr.pcm_length = sizeof(s_pcm_header) - 8;

    return  file;
}

int duer_store_voice_write(REC_FILE *file,const void *data, uint32_t size, int channel_id)
{
    if (file&&file->_file) {
        // 录音文件存储
        fwrite(data, 1, size, file->_file);
        file->_hdr.data_length += size;

        // PRINTF("size: %d\n", size);
	    // PRINTF("mono_size: %d\n", wav_buffers[channel_id]->mono_size);
	    // wav_buffers[channel_id]->mono_buffer = (int16_t *)malloc(size);
        // memcpy(wav_buffers[channel_id]->mono_buffer,data,size);
	// PRINTF("channel id: %d\n", channel_id);
        socket_send(data, size, channel_id);
        // wav_buffers[channel_id]->mono_size = size;
        // wav_buffers[channel_id]->_hdr.data_length += size;
	    // PRINTF("mono_size: %d\n", wav_buffers[channel_id]->mono_size);
    }
    
    return 0;
}

int duer_store_voice_end_nf(REC_FILE *file, int channel_id)
{
    if (file&&file->_file) {

        // wav_buffers[channel_id]->_hdr.pcm_length += s_pcm_header.data_length;

        // FILE    *__file;
        // char __name[64];
        // snprintf(__name, sizeof(__name), "./__channel-%d.wav", channel_id);
        // __file = fopen(__name, "wb");
	// PRINTF("monosize: %d\n", wav_buffers[channel_id]->mono_size);
	// PRINTF("%s  1\n", __name);
        // fwrite(&(wav_buffers[channel_id]->_hdr), 1, sizeof(wav_buffers[channel_id]->_hdr), __file);
	// PRINTF("%s  2\n", __name);
        // fwrite(wav_buffers[channel_id]->mono_buffer, 1, wav_buffers[channel_id]->mono_size, __file);
        // fclose(__file);
	// PRINTF("%s  3\n", __name);

        file->_hdr.pcm_length += s_pcm_header.data_length;
        fseek(file->_file, 0, SEEK_SET);
        // 修改文件开头的header
        fwrite(&file->_hdr, 1, sizeof(file->_hdr), file->_file);
        fclose(file->_file);
        file->_file = NULL;
    }
    return 0;
}


int duer_store_voice_end(REC_FILE *file, int channel_id)
{
    if (file&&file->_file) {

        // wav_buffers[channel_id]->_hdr.pcm_length += s_pcm_header.data_length;

        file->_hdr.pcm_length += s_pcm_header.data_length;
        fseek(file->_file, 0, SEEK_SET);
        // 修改文件开头的header
        fwrite(&file->_hdr, 1, sizeof(file->_hdr), file->_file);
        fclose(file->_file);
        file->_file = NULL;
    }
    return 0;
}

int read_pcm_mono_data(int16_t *in,int ilen,int16_t *out,int channel_cnt)
{
	int i=0;
	if((ilen%channel_cnt)){
			   PRINTF("invalid pcm data lenght!\n");
		       return -1;	
	 }
	 		
	 for(i=0;i<ilen/channel_cnt;i++){
			uint16_t mono_data = 0;
			int j=0;
			for(j=0;j<channel_cnt;j++){
				  out[i] += in[channel_cnt*i+j];
			}
			out[i] /= channel_cnt;
	}
    	
	  return ilen/channel_cnt;
}


int read_pcm_channel_data(int16_t *in,int ilen,int16_t *out,int channel_id,int channel_cnt)
{
        int i=0;
        
	    if((ilen%channel_cnt)){
		     PRINTF("invalid pcm data lenght!\n");
		     return -1;	
		}
		
		if(channel_id<0||channel_id>=channel_cnt){
		        PRINTF("please input invalid channel id \n");
		        return -1;	
		}
		
		for(i=0;i<ilen/channel_cnt;i++){
			out[i] = in[i*channel_cnt+channel_id];
		}
		
		return  ilen/channel_cnt;
}

static void   recording_pcm_data()
{
    int16_t *buffer = NULL;
    int16_t *mono_buffer = NULL;
    int mono_data_size = 0;
		
    snd_pcm_hw_params_get_period_size(s_index->params, &(s_index->frames), &(s_index->dir));
    if (s_index->frames < 0) {
        PRINTF("Get period size failed!");
        return;
    }
    
    PRINTF("frames %d dir %d\n",s_index->frames,s_index->dir);
    s_index->size = s_index->frames * FRAMES_SIZE;

    if (buffer) {
        free(buffer);
        buffer = NULL;
    }
	
    buffer = (int16_t *)malloc(s_index->size);
    if (!buffer) {
        PRINTF("malloc buffer failed!\n");
    } else {
        memset(buffer, 0, s_index->size);
    }

    mono_buffer = (int16_t *)malloc(s_index->size);
    if (!mono_buffer) {
        PRINTF("malloc buffer failed!\n");
    } else {
        memset(mono_buffer, 0, s_index->size);
    }
	
    while (1)
    {
        int i=0;
            int ret = snd_pcm_readi(s_index->handle, buffer, s_index->frames);
            
            if (ret == -EPIPE) {
                PRINTF("an overrun occurred!\n");
                snd_pcm_prepare(s_index->handle);
                continue;
            } else if (ret < 0) {
                PRINTF("read: %s\n", snd_strerror(ret));
            continue;
            } else if (ret != (int)s_index->frames) {
                PRINTF("read %d frames!\n", ret);
            continue;
            } else {
                // do nothing
            PRINTF("ret=%d %d\n",ret,s_index->size);
            }
        
        for(i=0;i<CHANNEL;i++){
            mono_data_size = read_pcm_channel_data(buffer,s_index->size>>1,mono_buffer,i,CHANNEL);
                duer_store_voice_write(s_rec_files[i],mono_buffer,mono_data_size<<1, i);
                // duer_store_voice_end_nf(s_rec_files[i], i);
        }
	// snd_pcm_close(s_index->handle);
        // return;
        // while(1){}
    }
    
    if (buffer) {
        free(buffer);
        buffer = NULL;
    }
    
    if(mono_buffer){
         free(mono_buffer);
	     mono_buffer=NULL;	
    }
	
    snd_pcm_drain(s_index->handle);
    snd_pcm_close(s_index->handle);
	
    if(s_index) {
        free(s_index);
        s_index = NULL;
    }
    	
	return;
}

static int duer_open_alsa_pcm()
{
    int ret = 0;
    
     s_index = (rec_config_t *)malloc(sizeof(rec_config_t));
    if (!s_index) {
	PRINTF("malloc fail\n");
        return -1;
    }
    
    memset(s_index, 0, sizeof(rec_config_t));
    s_index->frames = FRAMES_INIT;
    s_index->val = SAMPLE_RATE; // pcm sample rate
    
    int result = (snd_pcm_open(&(s_index->handle), PCM_STREAM_CAPTURE_DEVICE, SND_PCM_STREAM_CAPTURE, 0));
    if (result < 0){
        PRINTF("\n\n****unable to open pcm device: %s*********\n\n", snd_strerror(ret));
        ret = -1;
    }
    
    return ret;
}

static int duer_set_pcm_params()
{
    int ret = 0;
    
    snd_pcm_hw_params_alloca(&(s_index->params));
    snd_pcm_hw_params_any(s_index->handle, s_index->params);
    snd_pcm_hw_params_set_access(s_index->handle, s_index->params,
                                 SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(s_index->handle, s_index->params,
                                 SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(s_index->handle, s_index->params,
                                   CHANNEL);
    snd_pcm_hw_params_set_rate_near(s_index->handle, s_index->params,
                                    &(s_index->val), &(s_index->dir));
    snd_pcm_hw_params_set_period_size_near(s_index->handle, s_index->params,
                                           &(s_index->frames), &(s_index->dir));

    int result = snd_pcm_hw_params(s_index->handle, s_index->params);
    if (result < 0)    {
        PRINTF("unable to set hw parameters: %s\n", snd_strerror(result));
        ret = -1;
    }
    
    return ret;
}

void record_stop(int signo) 
{
     int i=0;
     
     PRINTF("oops! stop!!!\n");
     for(i=0;i<CHANNEL+1;i++){
	     if(s_rec_files[i] != NULL){
		       duer_store_voice_end(s_rec_files[i], i);
		 } 
	}
	
     _exit(0);
}

int main(int argc, char *argv[])
{
    int ret=0;
	int i;
        
	for(i=0;i<CHANNEL;i++){
	     s_rec_files[i] = NULL;  
	      s_rec_files[i] =  duer_store_voice_start(i);
	}
	
	signal(SIGINT, record_stop); 	
	PRINTF("%s %d\n",__FUNCTION__,__LINE__);    
	ret = duer_open_alsa_pcm();
        if (ret != 0) {
	        PRINTF("open pcm failed\n");
	        return -1;
	}
	
	PRINTF("%s %d\n",__FUNCTION__,__LINE__);    	
        ret = duer_set_pcm_params();
        if (ret != 0) {
	        PRINTF("duer_set_pcm_params failed\n");
		return -1;
	 }
	 
	 socket_conn();
	 
	 PRINTF("%s %d\n",__FUNCTION__,__LINE__); 
        recording_pcm_data();
	
	// struct dataPackage pkg;
	// pkg.size0 = wav_buffers[0]->mono_size;
	// pkg.size1 = wav_buffers[1]->mono_size;
	// pkg.size2 = wav_buffers[2]->mono_size;
	// pkg.size3 = wav_buffers[3]->mono_size;
	// pkg.data0 = wav_buffers[0]->mono_buffer;
	// pkg.data1 = wav_buffers[1]->mono_buffer;
	// pkg.data2 = wav_buffers[2]->mono_buffer;
	// pkg.data3 = wav_buffers[3]->mono_buffer;
	PRINTF("done\n");
	    
       return 0;
}

// 把所有的文件操作转化成对缓冲区的操作

int getData()
{
    PRINTF("c get data\n");
    int ret=0;
	int i;
        
	for(i=0;i<CHANNEL;i++){
	     s_rec_files[i] = NULL;  
	      s_rec_files[i] =  duer_store_voice_start(i);
	}
	
	signal(SIGINT, record_stop); 	
	PRINTF("%s %d\n",__FUNCTION__,__LINE__);    
	ret = duer_open_alsa_pcm();
        if (ret != 0) {
	        PRINTF("open pcm failed\n");
	        return -1;
	}
	
	PRINTF("%s %d\n",__FUNCTION__,__LINE__);    	
        ret = duer_set_pcm_params();
        if (ret != 0) {
	        PRINTF("duer_set_pcm_params failed\n");
		return -1;
	 }
	 
	 socket_conn();
	 
	 PRINTF("%s %d\n",__FUNCTION__,__LINE__); 
        recording_pcm_data();
	    
       return 0;
}

void socket_conn()
{
    PRINTF("socket conn\n");
    // 定义sockfd
    sock_cli = socket(AF_INET,SOCK_STREAM, 0);
 
    // 定义sockaddr_in
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(MYPORT);  // 服务器端口
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");  // 服务器ip
 
    // 连接服务器，成功返回0，错误返回-1
    if (connect(sock_cli, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        perror("connect");
        exit(1);
    }

}

void socket_send(const void *data, uint32_t size, int channel_id){
    // printf("socket send\n");
    send(sock_cli, data, size, 0);
}
