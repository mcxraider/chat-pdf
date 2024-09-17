"use client";

import React from "react";
import { TextHoverEffect } from "../components/home/text-hover-effect";
import { SparklesPreview } from "../components/home/sparkles";

const HomePage = () => {
  return (
    <div>
      <TextHoverEffect text="CHAT-PDF" duration={2} />
      <SparklesPreview />
    </div>
  );
};

export default HomePage;
