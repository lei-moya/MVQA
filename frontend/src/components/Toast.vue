<script setup>
import { ElMessage } from 'element-plus';

const props = defineProps({
  visible: Boolean,
  message: String
});

// 使用Element Plus的消息提示组件
if (props.visible && props.message) {
  ElMessage({
    message: props.message,
    type: 'info',
    duration: 3000
  });
}
</script>
