import History from "@/components/type/History";
import Screen from "@/components/type/Chat";
import Input from "@/components/type/Input";

export default function ChatTypePage() {
  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <History />
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <Screen />
        <Input />
      </div>
    </div>
  );
}
