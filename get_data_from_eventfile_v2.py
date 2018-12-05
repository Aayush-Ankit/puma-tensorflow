###############################################################################################################################################################Usage instructions
#python <script_name>.py <pathtoeventsfilefolder/eventsfile> <outputprocessedfilefolder/prefix_of_file> <full/ptrn1/sample> <samplevalue>
#Example Script
#Full Extraction of data
#-->python get_data_from_eventfile_v2.py simulationdata/puma_vgg16_slice_3bits_crsfreq_782/events.out.tfevents.1543380848.deepsim-12.labs.hpecorp.net extractedsimulationdata/puma_vgg16_crsfreq_782/3bits_slice full
### Aayush requested pattern first 2 epochs + middle 49,50,51 epochs + last 2 epochs
#-->python get_data_from_eventfile_v2.py simulationdata/puma_vgg16_slice_3bits_crsfreq_782/events.out.tfevents.1543380848.deepsim-12.labs.hpecorp.net extractedsimulationdata/sample_test/3bits_slice ptrn1
## Regular sampling of epochs
#-->python get_data_from_eventfile_v2.py simulationdata/puma_vgg16_slice_3bits_crsfreq_782/events.out.tfevents.1543380848.deepsim-12.labs.hpecorp.net extractedsimulationdata/sample_test/3bits_slice sample 5
#############################################################################################################################################################

import tensorflow as tf
import sys
events_file_path=sys.argv[1]
out_file_path_prefix=sys.argv[2]
sample_step=sys.argv[3]

if len(sys.argv)>4:
   sample_epochs=int(sys.argv[4])
   sample_step+="_"+sys.argv[4]+'epochs'
loss_out_file_path=out_file_path_prefix+'_loss_sample_'+sample_step+'.log'
parallel_write_sat_out_file_path=out_file_path_prefix+'_parallel_write_saturation_sample_'+sample_step+'.log'
parallel_write_sat_slice0_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice0_sample_'+sample_step+'.log'
parallel_write_sat_slice1_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice1_sample_'+sample_step+'.log'
parallel_write_sat_slice2_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice2_sample_'+sample_step+'.log'
parallel_write_sat_slice3_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice3_sample_'+sample_step+'.log'
parallel_write_sat_slice4_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice4_sample_'+sample_step+'.log'
parallel_write_sat_slice5_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice5_sample_'+sample_step+'.log'
parallel_write_sat_slice6_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice6_sample_'+sample_step+'.log'
parallel_write_sat_slice7_out_file_path=out_file_path_prefix+'_parallel_write_saturation_slice7_sample_'+sample_step+'.log'
train_accuracy_top_1_out_file_path=out_file_path_prefix+'_train_accuracy_top_1_sample_'+sample_step+'.log'
train_accuracy_top_5_out_file_path=out_file_path_prefix+'_train_accuracy_top_5_sample_'+sample_step+'.log'
validation_accuracy_top_1_out_file_path=out_file_path_prefix+'_validation_accuracy_top_1_sample_'+sample_step+'.log'
validation_accuracy_top_5_out_file_path=out_file_path_prefix+'_validation_accuracy_top_5_sample_'+sample_step+'.log'
file_loss=open(loss_out_file_path,"w")
file_pws=open(parallel_write_sat_out_file_path,"w")
file_pws_s0=open(parallel_write_sat_slice0_out_file_path,"w")
file_pws_s1=open(parallel_write_sat_slice1_out_file_path,"w")
file_pws_s2=open(parallel_write_sat_slice2_out_file_path,"w")
file_pws_s3=open(parallel_write_sat_slice3_out_file_path,"w")
file_pws_s4=open(parallel_write_sat_slice4_out_file_path,"w")
file_pws_s5=open(parallel_write_sat_slice5_out_file_path,"w")
file_pws_s6=open(parallel_write_sat_slice6_out_file_path,"w")
file_pws_s7=open(parallel_write_sat_slice7_out_file_path,"w")
file_tacc_top1=open(train_accuracy_top_1_out_file_path,"w")
file_tacc_top5=open(train_accuracy_top_5_out_file_path,"w")
file_vacc_top1=open(validation_accuracy_top_1_out_file_path,"w")
file_vacc_top5=open(validation_accuracy_top_5_out_file_path,"w")
if sample_step=="full":
   sample_step_not_used=1
   sample_ptr1=0
elif sample_step=="ptrn1":
   sample_ptr1=1
   sample_step_not_used=0
elif sample_step > 0:
   sample_step_not_used=0;
   sample_ptr1=0;
stepcount=0
epochcount=0
samplecount=0
for e in tf.train.summary_iterator(events_file_path):
 if sample_step_not_used==1:
    for v in e.summary.value:
        if v.tag == 'Loss':
            file_loss.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation":
            file_pws.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_0":
            file_pws_s0.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_1":
            file_pws_s1.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_2":
            file_pws_s2.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_3":
            file_pws_s3.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_4":
            file_pws_s4.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_5":
            file_pws_s5.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_6":
            file_pws_s6.write(str(v.simple_value)+"\n")
        elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_7":
            file_pws_s7.write(str(v.simple_value)+"\n")
        elif v.tag == "Training_accuracy_-_Top-1":
            file_tacc_top1.write(str(v.simple_value)+"\n")
        elif v.tag == "Training_accuracy_-_Top-5":
            file_tacc_top5.write(str(v.simple_value)+"\n")
        elif v.tag == "Validation Accuracy - Top-1":
            file_vacc_top1.write(str(v.simple_value)+"\n")
        elif v.tag == "Validation Accuracy - Top-5":
            file_vacc_top5.write(str(v.simple_value)+"\n")
 elif sample_ptr1==1:
    stepcount+=1
    steps_per_epoch=782
    if (stepcount < 2*steps_per_epoch) or (49*steps_per_epoch < stepcount < 51*steps_per_epoch) or (stepcount > 98*steps_per_epoch):
       #print(str(stepcount))
       for v in e.summary.value:
         if v.tag == 'Loss':
            file_loss.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation":
            file_pws.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_0":
            file_pws_s0.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_1":
            file_pws_s1.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_2":
            file_pws_s2.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_3":
            file_pws_s3.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_4":
            file_pws_s4.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_5":
            file_pws_s5.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_6":
            file_pws_s6.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_7":
            file_pws_s7.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Training_accuracy_-_Top-1":
            file_tacc_top1.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Training_accuracy_-_Top-5":
            file_tacc_top5.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Validation Accuracy - Top-1":
            file_vacc_top1.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Validation Accuracy - Top-5":
            file_vacc_top5.write(str(stepcount)+","+str(v.simple_value)+"\n")
 else:
    stepcount+=1
    steps_per_epoch=782
#    sample_epochs=int(sys.argv[4])
    if stepcount%steps_per_epoch==0:
       epochcount+=1
       if samplecount!=0:
          samplecount-=1
       else:
          samplecount=sample_epochs
   

    #if (stepcount < steps_per_epoch) or (49*steps_per_epoch < stepcount < 51*steps_per_epoch) or (stepcount > 98*steps_per_epoch):
    if epochcount == 0 or samplecount==0: 
      #print(str(stepcount))
       for v in e.summary.value:
         if v.tag == 'Loss':
            file_loss.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation":
            file_pws.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_0":
            file_pws_s0.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_1":
            file_pws_s1.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_2":
            file_pws_s2.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_3":
            file_pws_s3.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_4":
            file_pws_s4.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_5":
            file_pws_s5.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_6":
            file_pws_s6.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "PUMA_Parallel_Write_Saturation-Slice_7":
            file_pws_s7.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Training_accuracy_-_Top-1":
            file_tacc_top1.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Training_accuracy_-_Top-5":
            file_tacc_top5.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Validation Accuracy - Top-1":
            file_vacc_top1.write(str(stepcount)+","+str(v.simple_value)+"\n")
         elif v.tag == "Validation Accuracy - Top-5":
            file_vacc_top5.write(str(stepcount)+","+str(v.simple_value)+"\n")
file_pws.close()
file_pws_s0.close()
file_pws_s1.close()
file_pws_s2.close()
file_pws_s3.close()
file_pws_s4.close()
file_pws_s5.close()
file_pws_s6.close()
file_pws_s7.close()
file_tacc_top1.close()
file_tacc_top5.close()
file_vacc_top1.close()
file_vacc_top5.close()
