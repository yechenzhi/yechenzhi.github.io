---
layout: page
title: Part I — Classical CUDA SGEMM
permalink: /gpu-kernels/gemm/classical-cuda-sgemm/
nav: false
series: classical-cuda-sgemm
---

[← GEMM]({{ '/gpu-kernels/gemm/' | relative_url }})

{% assign series_posts = site.posts | where: 'series', page.series | sort: 'episode' %}
{% if series_posts.size > 0 %}
  <ol>
    {% for post in series_posts %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ol>
{% endif %}
