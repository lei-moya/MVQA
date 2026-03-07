<template>
  <div v-if="show" class="input-dialog-overlay">
    <div class="input-dialog">
      <div class="dialog-header">
        <h3><i class="fa-solid fa-edit"></i> 视频信息</h3>
        <button @click="close" class="close-btn">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>
      <div class="dialog-body">
        <div class="input-box">
          <label>视频标题</label>
          <input
            type="text"
            v-model="title"
            class="input-control"
            placeholder="请输入视频标题..."
            ref="titleInput"
          >
        </div>
        <div class="input-box">
          <label>视频简介</label>
          <textarea
            v-model="description"
            class="input-control textarea"
            placeholder="请输入视频简介..."
            rows="4"
          ></textarea>
        </div>
      </div>
      <div class="dialog-footer">
        <button @click="close" class="btn btn-outline">取消</button>
        <button @click="submit" class="btn btn-primary">
          <i class="fa-solid fa-check"></i> 提交
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue';

const props = defineProps({
  show: Boolean,
  files: Array
});

const emit = defineEmits(['close', 'submit']);

const title = ref('');
const description = ref('');
const titleInput = ref(null);

const close = () => {
  emit('close');
  title.value = '';
  description.value = '';
};

const submit = async () => {
  if (!title.value.trim() || !description.value.trim()) {
    return;
  }

  emit('submit', {
    title: title.value,
    description: description.value,
    files: props.files
  });

  close();
};

onMounted(() => {
  nextTick(() => {
    if (titleInput.value) {
      titleInput.value.focus();
    }
  });
});
</script>
